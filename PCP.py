def node_wait_next_followup(state: PCPState) -> PCPState:
    """
    Keeps the conversation alive after showing address/warning/etc.
    Interrupts to wait for the next user message, then stores it into csr_query.
    """
    requested = interrupt({
        "prompt": "",                 # IMPORTANT: empty prompt; UI will display prior ai_response
        "stage": "WAIT_NEXT_FOLLOWUP"
    })

    state["csr_query"] = str(requested)
    return state

############################################################

def route_after_provider_followup(state: PCPState) -> str:
    if state.get("last_followup_action") == "assign_pcp":
        return "assign"
    return "loop"
###########################################################

elif stage_from_node == "WAIT_NEXT_FOLLOWUP":
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage=last_state.get("stage", ""),
            ai_response=last_state.get("ai_response", ""),
            csr_query=req.message or "",
            prompts=last_state.get("prompts", []),
            prompt_title=last_state.get("prompt_title", ""),
            ai_response_code=last_state.get("ai_response_code", 101),
            ai_response_type=last_state.get("ai_response_type", "AURA"),
        )
    )
