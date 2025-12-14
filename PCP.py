def interrupt_with_source(state: PCPState, payload: Dict[str, Any]):
    # attach call source to the interrupt payload so FastAPI can trust it
    payload["call_type"] = state.get("call_type") or ""
    payload["call_name"] = state.get("call_name") or ""
    return interrupt(payload)

========================================================================================

call_source = interrupt_payload.get("call_name") or last_state.get("call_name") or ""

