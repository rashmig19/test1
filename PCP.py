def _set_call_source(state: PCPState, call_type: str, call_name: str) -> None:
    state["call_type"] = call_type
    state["call_name"] = call_name

def mark_llm(state: PCPState) -> None:
    # configurable label (not hardcoded)
    llm_label = getattr(settings, "CALL_SOURCE_LLM_LABEL", "LLM")
    _set_call_source(state, "LLM", llm_label)

def mark_api(state: PCPState, fn_or_name: Any) -> None:
    # use function name when possible (no hardcoded API names)
    name = getattr(fn_or_name, "__name__", None) or str(fn_or_name)
    _set_call_source(state, "API", name)

def mark_system(state: PCPState) -> None:
    sys_label = getattr(settings, "CALL_SOURCE_SYSTEM_LABEL", "SYSTEM")
    _set_call_source(state, "SYSTEM", sys_label)

#################################################################################

