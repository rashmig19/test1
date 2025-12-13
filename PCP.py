def node_return_to_menu(state: PCPState) -> PCPState:
    """
    Emits the START menu payload (AIResponse empty) and interrupts.
    This is used when user says YES after Specialist inquiry completion.
    """
    logger.debug("node_return_to_menu")

    # Pull from config when available (avoid hardcoding text in code)
    menu_title = getattr(settings, "MENU_PROMPT_TITLE", "How can I assist you today?")
    menu_prompts = getattr(settings, "MENU_PROMPTS", None)
    if not isinstance(menu_prompts, list) or not menu_prompts:
        menu_prompts = DEFAULT_PROMPTS

    # Exact UI payload
    state["ai_response"] = ""
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = menu_title
    state["prompts"] = menu_prompts
    state["stage"] = "START"

    # Interrupt so /chat returns immediately with menu
    requested = interrupt({
        "prompt": "",
        "stage": "RETURN_TO_MENU",
        "prompt_title": menu_title,
        "prompts": menu_prompts,
        "ai_response_code": 101,
        "ai_response_type": "AURA",
    })

    # Resume value becomes the next menu selection
    state["csr_query"] = str(requested).strip()
    return state

##################################

def node_specialist_post_completion(state: PCPState) -> PCPState:
    """
    After specialist address shown with Yes/No prompts:
    - Yes -> route to RETURN_TO_MENU node
    - No  -> closing message, end
    """
    logger.debug("node_specialist_post_completion")

    txt = (state.get("csr_query") or "").strip()

    # Use Horizon to interpret yes/no semantically
    sys = (
        "Classify if the user's response means YES or NO.\n"
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

    # YES (or unknown -> treat as YES to keep user moving)
    # reset specialist-only fields but keep member/thread
    state["providers_result"] = None
    state["csr_query"] = ""
    state["stage"] = "GO_MENU"
    return state

###########################################33

builder.add_node("return_to_menu", node_return_to_menu)

def route_after_specialist_completion(state: PCPState) -> str:
    if state.get("stage") == "CLOSED":
        return "end"
    if state.get("stage") == "GO_MENU":
        return "menu"
    return "menu"


builder.add_conditional_edges(
    "specialist_post_completion",
    path=route_after_specialist_completion,
    path_map={
        "menu": "return_to_menu",
        "end": END,
    },
)


#################################

elif stage_from_node == "RETURN_TO_MENU":
    ptitle = interrupt_payload.get("prompt_title") or last_state.get("prompt_title") or getattr(settings, "MENU_PROMPT_TITLE", "How can I assist you today?")
    pr = interrupt_payload.get("prompts") or last_state.get("prompts") or DEFAULT_PROMPTS

    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="START",                 # ✅ as you requested
            ai_response="",                # ✅ empty
            csr_query=req.message or "",
            prompts=pr,
            prompt_title=ptitle,
            ai_response_code=101,
            ai_response_type="AURA",
        )
    )

