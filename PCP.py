    if telemetry is not None:
        telemetry["llm_latency_ms"] = telemetry.get("llm_latency_ms", 0.0) + ((perf_counter() - t0) * 1000.0)
