def node_specialist_provider_address(state: PCPState) -> PCPState:
    """
    After specialist provider grid is shown, user selects a provider id and asks for address.
    Return address JSON in AIResponse, and mark conversation as completed (inquiry only).
    """
    logger.debug("node_specialist_provider_address")

    providers = state.get("providers_result") or []
    if not providers:
        state["ai_response"] = "No providers are available in the current result. Please start a new search."
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 404
        state["ai_response_type"] = "AURA"
        state["stage"] = "ERROR"
        return state

    # We should already have the user's message in csr_query (resume value)
    user_msg = (state.get("csr_query") or "").strip()
    if not user_msg:
        requested = interrupt({"prompt": "", "stage": "WAIT_SPECIALIST_PROVIDER_ACTION"})
        state["csr_query"] = str(requested).strip()
        user_msg = state["csr_query"]

    # Use existing LLM helper to understand intent + provider id
    decision = llm_decide_followup_action(user_msg, providers)
    action = decision.get("action")
    pid = decision.get("provider_id")

    # Specialist flow supports ONLY address inquiry (not assignment)
    if action != "address" or not pid:
        state["ai_response"] = (
            "Please provide a Provider ID and ask for its address (for example: 'address of 12345678')."
        )
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 110
        state["ai_response_type"] = "AURA"
        state["stage"] = "SPECIALIST_NEED_ADDRESS_CLARIFICATION"
        return state

    # Find chosen provider from providers_result
    chosen = None
    for p in providers:
        cand = (
            p.get("providerId")
            or p.get("provId")
            or p.get("ProviderID")
            or p.get("id")
        )
        if str(cand).strip() == str(pid).strip():
            chosen = p
            break

    if not chosen:
        state["ai_response"] = "I couldn't find that Provider ID in the current results. Please pick a Provider ID from the list."
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 110
        state["ai_response_type"] = "AURA"
        state["stage"] = "SPECIALIST_NEED_ADDRESS_CLARIFICATION"
        return state

    prov_id = str(chosen.get("providerId") or pid or "").strip()
    addr_payload = {
        "providerAddress": [{
            "ProviderID": prov_id,
            "AddressType": chosen.get("addressType"),
            "AddressLine1": chosen.get("address1") or chosen.get("addressLine1"),
            "AddressLine2": chosen.get("address2") or chosen.get("addressLine2"),
            "City": chosen.get("city"),
            "State": chosen.get("state"),
            "Zip": chosen.get("zip") or chosen.get("zipCode"),
            "Country": chosen.get("country") or chosen.get("county"),
            "Phone": chosen.get("phone"),
            "Fax": chosen.get("fax"),
            "EffectiveDate": chosen.get("effectiveDate"),
            "TerminationDate": chosen.get("terminationDate"),
        }]
    }

    # Final inquiry response (completed)
    state["ai_response"] = json.dumps(addr_payload, default=str, separators=(",", ":"))
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 110
    state["prompt_title"] = "Do you need further assistance?"
    state["prompts"] = ["Yes", "No"]
    state["stage"] = "COMPLETED"

    requested = interrupt({
        "prompt": state["ai_response"],
        "stage": "SPECIALIST_COMPLETED",
        "prompt_title": state["prompt_title"],
        "prompts": state["prompts"],
        "ai_response_code": state["ai_response_code"],
        "ai_response_type": state["ai_response_type"],
    })

    # store next yes/no answer for routing
    state["csr_query"] = str(requested).strip()
    return state
=====================================
def node_specialist_post_completion(state: PCPState) -> PCPState:
    """
    After specialist address is shown with Yes/No prompts, route:
    - Yes -> reset and go back to menu
    - No  -> closing message + END
    """
    logger.debug("node_specialist_post_completion")

    txt = (state.get("csr_query") or "").strip().lower()

    # Let Horizon interpret yes/no semantically (no hardcoded matching only)
    sys = (
        "You classify whether the user response means YES or NO.\n"
        "Return ONLY JSON: {\"answer\":\"yes\"|\"no\"|\"unknown\"}\n"
    )
    raw = call_horizon(sys, txt).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        i = raw.find("{")
        if i != -1:
            raw = raw[i:]
    ans = "unknown"
    try:
        ans = (json.loads(raw).get("answer") or "unknown").lower()
    except Exception:
        ans = "unknown"

    if ans == "no":
        state["ai_response"] = "We're closing your request-feel free to return if you need anything else."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 101
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "CLOSED"
        return state

    # YES or unknown -> send to menu (same behavior as start)
    # reset specialist-specific fields (keep member details/thread)
    state["providers_result"] = None
    state["specialist_service_specialty"] = None
    state["specialist_raw_filter_input"] = None
    state["csr_query"] = ""   # menu will interrupt and refill
    state["stage"] = "MENU"
    return state
==============================================
builder.add_node("specialist_provider_address", node_specialist_provider_address)
builder.add_node("specialist_post_completion", node_specialist_post_completion)
=========================================================
builder.add_edge("run_specialist_generic_search", "specialist_provider_address")
builder.add_edge("specialist_provider_address", "specialist_post_completion")
================================================================
def route_after_specialist_completion(state: PCPState) -> str:
    if state.get("stage") == "CLOSED":
        return "end"
    return "menu"
============================
builder.add_conditional_edges(
    "specialist_post_completion",
    path=route_after_specialist_completion,
    path_map={
        "menu": "start",
        "end": END,
    },
)
==========================================
elif stage_from_node == "SPECIALIST_COMPLETED":
    # The interrupt prompt is the address JSON.
    ai_resp_text = prompt or last_state.get("ai_response", "")
    ptitle = interrupt_payload.get("prompt_title") or last_state.get("prompt_title") or "Do you need further assistance?"
    pr = interrupt_payload.get("prompts") or last_state.get("prompts") or ["Yes", "No"]

    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="COMPLETED",                 # ✅ as you requested
            ai_response=ai_resp_text,
            csr_query=req.message or "",
            prompts=pr,
            prompt_title=ptitle,
            ai_response_code=110,
            ai_response_type="AURA",
        )
    )
====================================================
