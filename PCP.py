# ---------------- Logfire (safe optional) ----------------
try:
    import logfire  # optional dependency
    logfire.configure()
    logfire.instrument_fastapi(app)
    LOGFIRE_ENABLED = True
except Exception as logfire_exc:
    LOGFIRE_ENABLED = False
    logger.warning("Logfire disabled: %s", logfire_exc)

    class _LogfireNoop:
        @staticmethod
        def info(event_name: str, **fields: Any):
            logger.info("logfire noop %s %s", event_name, fields)

    logfire = _LogfireNoop()
