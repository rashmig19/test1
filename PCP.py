    # Specialist flow
    flow: Optional[str]  # "pcp" | "specialist"
    specialist_service_specialty: Optional[str]  # e.g. "S204"

###############################################################33

SPECIALIST_SERVICE_QUESTION = getattr(
    settings,
    "SPECIALIST_SERVICE_QUESTION",
    "<p>What services are you looking for at this location?</p>"
)


###############################################################

def llm_route_menu_intent(user_text: str) -> str:
    """
    Returns: "assign_pcp" | "specialist" | "unsupported"
    Uses Horizon LLM so we can handle spelling mistakes and semantic variants.
    """
    system = (
        "You are an intent classifier for a health plan CSR assistant.\n"
        "Classify the user's menu choice into one of:\n"
        '- "assign_pcp"\n'
        '- "specialist"\n'
        '- "unsupported"\n'
        "Consider spelling mistakes and semantic variants.\n"
        "Return ONLY JSON like: {\"intent\":\"assign_pcp\"}\n"
    )
    raw = call_horizon(system, user_text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        data = json.loads(raw)
        intent = str(data.get("intent") or "").strip()
        if intent in ("assign_pcp", "specialist", "unsupported"):
            return intent
        return "unsupported"
    except Exception:
        logger.warning("Failed to parse menu intent JSON from LLM: %s", raw)
        return "unsupported"


#######################################################################

def node_specialist_ask_service(state: PCPState) -> PCPState:
    """
    Specialist flow step 1:
    Ask exact question (AIResponse must be EXACT HTML),
    then interrupt and store the service specialty code.
    """
    logger.debug("node_specialist_ask_service")

    # mark flow
    state["flow"] = "specialist"

    # Ask Horizon to output EXACT text, then enforce exact output safely
    system = (
        "You are a CSR assistant. You MUST return EXACTLY the following text and nothing else:\n"
        f"{SPECIALIST_SERVICE_QUESTION}"
    )
    ai_msg = call_horizon(system, "Return the exact text now.").strip()

    # Enforce exactness (requirement says exact string)
    if ai_msg != SPECIALIST_SERVICE_QUESTION:
        ai_msg = SPECIALIST_SERVICE_QUESTION

    state["ai_response"] = ai_msg
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 109
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "ASK_SPECIALIST_SERVICE"

    requested = interrupt({
        "prompt": ai_msg,
        "stage": state["stage"],
    })

    # On resume: store the specialty code for later steps
    state["specialist_service_specialty"] = str(requested).strip()
    state["csr_query"] = ""  # keep clean
    state["stage"] = "SPECIALIST_SERVICE_CAPTURED"

    # Nothing else yet (next steps will be added later)
    state["ai_response"] = ""
    state["prompt_title"] = ""
    state["prompts"] = []
    state["ai_response_code"] = 101
    state["ai_response_type"] = "AURA"
    return state


###############################################################
def route_from_menu(state: PCPState) -> str:
    text = (state.get("csr_query") or "").strip()
    if not text:
        return "unsupported"
    return llm_route_menu_intent(text)


###############################################################

builder.add_node("specialist_ask_service", node_specialist_ask_service)


builder.add_conditional_edges(
    "start",
    path=route_from_menu,
    path_map={
        "assign_pcp": "assign_pcp_ask_termination",
        "specialist": "specialist_ask_service",
        "unsupported": END,
    },
)


###############################################3

builder.add_edge("specialist_ask_service", END)


##############33
            elif stage_from_node == "ASK_SPECIALIST_SERVICE":
                ai_resp_text = prompt or last_state.get("ai_response", "") or SPECIALIST_SERVICE_QUESTION
                # ensure exact
                if ai_resp_text != SPECIALIST_SERVICE_QUESTION:
                    ai_resp_text = SPECIALIST_SERVICE_QUESTION

                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage="ASK_SPECIALIST_SERVICE",
                        ai_response=ai_resp_text,
                        csr_query=req.message or "",
                        prompts=[],
                        prompt_title="",
                        ai_response_code=109,
                        ai_response_type="AURA",
                    )
                )


##################################################

