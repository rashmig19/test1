if action == "provider_id_only":
    state["ai_response"] = "Please check the response that you have provided"
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "SPECIALIST_PROVIDER_FOLLOWUP_RESPONSE"
    state["last_followup_action"] = "provider_id_only"
    return state
