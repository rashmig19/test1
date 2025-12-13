elif stage_from_node == "SPECIALIST_PROVIDER_FOLLOWUP_RESPONSE":
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="SPECIALIST_PROVIDER_FOLLOWUP_RESPONSE",
            ai_response=prompt or last_state.get("ai_response", ""),
            csr_query=req.message or "",
            prompts=[],
            prompt_title="",
            ai_response_code=101,
            ai_response_type="AURA",
        )
    )
