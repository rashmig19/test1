    request_start = perf_counter()
    telemetry: Dict[str, Any] = {
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_latency_ms": 0.0,
        "thread_id": req.thread_id,
    }
    response_payload: Dict[str, Any] = {}
    error: Optional[Exception] = None

    set_llm_telemetry(telemetry)
