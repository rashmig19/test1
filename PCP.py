    if isinstance(content, str):
        out_text = content.strip()
        if telemetry is not None:
            telemetry["llm_completion_tokens"] = telemetry.get("llm_completion_tokens", 0) + estimate_tokens(out_text)
            telemetry["llm_latency_ms"] = telemetry.get("llm_latency_ms", 0.0) + ((perf_counter() - t0) * 1000.0)
        return out_text
