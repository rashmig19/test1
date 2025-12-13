# Ask EXACT question via Horizon but enforce exact final sentence
    # (You can keep Horizon for "LLM requirement", but hard-guard the final text)
    desired = "<p>What services are you looking for at this location?</p>"
    _ = call_horizon(
        "You are a CSR assistant. Ask exactly: <p>What services are you looking for at this location?</p> "
        "Return exactly that string.",
        "Ask the question."
    )
    ai_msg = desired  # enforce exact required text

    state["ai_response"] = ai_msg
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 109
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "ASK_SERVICE_SPECIALITY"

    requested = interrupt({"prompt": ai_msg, "stage": state["stage"]})
    state["service_speciality"] = str(requested).strip()
    state["csr_query"] = ""  # clear so next node doesn't reuse it
    return state
