from contextvars import ContextVar
from typing import Optional, Dict, Any


_llm_telemetry: ContextVar[Optional[Dict[str, Any]]] = ContextVar("llm_telemetry", default=None)


def set_llm_telemetry(telemetry: Optional[Dict[str, Any]]):
    """Expose telemetry context so FastAPI handler can collect per-request LLM stats."""
    _llm_telemetry.set(telemetry)

def get_llm_telemetry() -> Optional[Dict[str, Any]]:
    return _llm_telemetry.get()
