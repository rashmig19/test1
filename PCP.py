    # If nothing set inside interrupts, build completed payload
    if not response_payload:
        response_payload = base_response(
            thread_id=req.thread_id,
            stage=last_state.get("stage", ""),
            ai_response=last_state.get("ai_response", ""),
            csr_query=req.message or "",
            prompts=last_state.get("prompts", []),
            prompt_title=last_state.get("prompt_title", ""),
            ai_response_code=last_state.get("ai_response_code", 101),
            ai_response_type=last_state.get("ai_response_type", "AURA"),
            call_source=call_source or last_state.get("call_name") or "",
        )

    # ---------------- metrics finalize (never break API) ----------------
    try:
        total_latency_ms = round((perf_counter() - request_start) * 1000, 2)

        response_text = ""
        if isinstance(response_payload, dict):
            response_text = str(response_payload.get("AIResponse", ""))

        token_usage = {
            "request_tokens": estimate_tokens(req.message),
            "response_tokens": estimate_tokens(response_text),
            "llm_prompt_tokens": telemetry.get("llm_prompt_tokens", 0),
            "llm_completion_tokens": telemetry.get("llm_completion_tokens", 0),
        }
        token_usage["llm_total_tokens"] = token_usage["llm_prompt_tokens"] + token_usage["llm_completion_tokens"]
        token_usage["total_tokens_all"] = (
            token_usage["request_tokens"] + token_usage["response_tokens"] + token_usage["llm_total_tokens"]
        )

        # Update per-session totals
        token_usage_totals: Dict[str, float] = {}
        tid = req.thread_id
        if tid:
            with token_totals_lock:
                totals = SESSION_TOKEN_TOTALS.get(tid) or _blank_token_totals()
                totals["request_tokens"] += token_usage["request_tokens"]
                totals["response_tokens"] += token_usage["response_tokens"]
                totals["llm_prompt_tokens"] += token_usage["llm_prompt_tokens"]
                totals["llm_completion_tokens"] += token_usage["llm_completion_tokens"]
                totals["llm_total_tokens"] += token_usage["llm_total_tokens"]
                totals["total_tokens_all"] += token_usage["total_tokens_all"]
                SESSION_TOKEN_TOTALS[tid] = totals
                token_usage_totals = dict(totals)

        # Attach metrics into API response
        if isinstance(response_payload, dict):
            response_payload["LatencyMs"] = total_latency_ms
            response_payload["LLMLatencyMs"] = round(float(telemetry.get("llm_latency_ms", 0.0)), 2)
            response_payload["TokenUsage"] = token_usage
            if token_usage_totals:
                response_payload["TokenUsageTotals"] = token_usage_totals

        # Logfire event (works even if noop)
        logfire.info(
            "chat_request",
            path="/chat",
            latency_ms=total_latency_ms,
            llm_latency_ms=round(float(telemetry.get("llm_latency_ms", 0.0)), 2),
            error=str(error) if error else None,
            **token_usage,
            **({f"total_{k}": v for k, v in token_usage_totals.items()} if token_usage_totals else {}),
        )

        # Persist metrics (CSV -> XLSX best effort)
        with metrics_lock:
            total_in = token_usage["request_tokens"] + token_usage["llm_prompt_tokens"]
            total_out = token_usage["response_tokens"] + token_usage["llm_completion_tokens"]
            api_hit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ai_response = str(response_payload.get("AIResponse", "")) if isinstance(response_payload, dict) else ""
            prompt_title = str(response_payload.get("PromptTitle", "")) if isinstance(response_payload, dict) else ""

            MAX_CELL_LEN = int(getattr(settings, "METRICS_MAX_CELL_LEN", 32700))
            if len(ai_response) > MAX_CELL_LEN:
                ai_response = ai_response[:MAX_CELL_LEN]
            if len(prompt_title) > MAX_CELL_LEN:
                prompt_title = prompt_title[:MAX_CELL_LEN]

            _append_metrics_row([
                tid,
                req.message or "",
                ai_response,
                prompt_title,
                total_in,
                total_out,
                token_usage["total_tokens_all"],
                total_latency_ms,
                api_hit_time,
            ])

    except Exception as exc:
        logger.warning("Failed to finalize metrics: %s", exc)

    finally:
        set_llm_telemetry(None)

    return JSONResponse(response_payload)
