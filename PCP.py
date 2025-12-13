def node_provider_id_only_warning(state: PCPState) -> PCPState:
    """
    If user sends only Provider ID without asking address/assign,
    show the warning message and STOP here (interrupt).
    Next message resumes back to provider_interaction.
    """
    msg = "Please check the response that you have provided."

    state["ai_response"] = msg
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "PROVIDER_ID_ONLY_WARNING"

    # IMPORTANT: interrupt each time (no fall-through)
    requested = interrupt({
        "prompt": msg,
        "stage": state["stage"],
    })

    # capture the next user input so provider_interaction can process it
    state["csr_query"] = str(requested).strip()
    return state


#####################################

elif stage_from_node == "PROVIDER_ID_ONLY_WARNING":
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="PROVIDER_ID_ONLY_WARNING",
            ai_response=prompt or last_state.get("ai_response", ""),
            csr_query=req.message or "",
            prompts=[],
            prompt_title="",
            ai_response_code=101,
            ai_response_type="AURA",
        )
    )
