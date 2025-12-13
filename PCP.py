    specialist_raw_filter_input: Optional[str]

###########################################################3

SPECIALIST_FILTERS_TEMPLATE = getattr(
    settings,
    "SPECIALIST_FILTERS_TEMPLATE",
    "<p><b>Ask the below questions:</b></p>\n"
    "<ul>\n"
    "<li>May I know any other preferred Language, if not English and Gender (Male/Female/Either)?</li>\n"
    "<li>The default distance is <<default_distance>> miles, please confirm if any changes needed.</li>\n"
    "<li>Search would be performed based on member's home address. If you would like to search provider at different location,please provide with zip code and address line 1.</li>\n"
    "<li>Search will be performed with today’s date unless a different date is provided (MM-DD-YYYY).</li>\n"
    "</ul>"
)


#############################################################

def node_specialist_ask_filters(state: PCPState) -> PCPState:
    """
    After specialist service specialty code is captured:
    - compute default distance from CSV using PractitionerType="Specialist" and group_id
    - ask Horizon to output the exact HTML template with default_distance replaced
    - AIResponseType=Dialog, AIResponseCode=103, PromptTitle/Prompts empty
    - interrupt to capture user filter inputs for next step (later)
    """
    logger.debug("node_specialist_ask_filters")

    # Ensure we have member market identifiers (group_id/subscriber_id) for CSV + later API calls
    member_id = (state.get("member_id") or "").strip()
    if not member_id:
        state["ai_response"] = "Member information is missing. Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    if not (state.get("group_id") and state.get("subscriber_id") and state.get("meme_ck") and state.get("grgr_ck")):
        try:
            member_response = member_search(dob="", mbrId=member_id, firstNm="", lastNm="")
            if isinstance(member_response, list):
                if not member_response:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
                member_payload = member_response[0]
            else:
                member_payload = member_response

            state["grgr_ck"] = str(member_payload.get("grgrCk") or member_payload.get("sbsbCk") or "").strip()
            state["meme_ck"] = str(member_payload.get("memeCk") or "").strip()
            state["group_id"] = str(member_payload.get("grpId") or "").strip()
            state["subscriber_id"] = str(member_payload.get("subscriberId") or "").strip()
        except Exception as ex:
            logger.exception("member_search failed in specialist flow: %s", ex)
            state["ai_response"] = "Unable to fetch member details right now. Please try again later."
            state["ai_response_type"] = "AURA"
            state["ai_response_code"] = 500
            state["prompt_title"] = ""
            state["prompts"] = []
            state["stage"] = "ERROR"
            return state

    group_id = (state.get("group_id") or "").strip()
    if not group_id:
        state["ai_response"] = "Member group information is missing. Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    # Default distance from CSV for Specialist
    try:
        default_distance = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")
    except Exception as ex:
        logger.exception("Default distance lookup failed: %s", ex)
        state["ai_response"] = "Unable to determine the default distance right now. Please try again later."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    expected_html = SPECIALIST_FILTERS_TEMPLATE.replace("<<default_distance>>", str(default_distance))

    # Ask Horizon to produce EXACT HTML (then enforce exactness to meet UI contract)
    system = (
        "You are a CSR assistant.\n"
        "Return EXACTLY the following HTML (no extra spaces, no explanation, no markdown fences):\n"
        f"{expected_html}"
    )
    ai_msg = call_horizon(system, "Return the exact HTML now.").strip()
    if ai_msg != expected_html:
        ai_msg = expected_html  # enforce contract

    state["ai_response"] = ai_msg
    state["ai_response_type"] = "Dialog"
    state["ai_response_code"] = 103
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "ASK_SPECIALIST_FILTERS"

    requested = interrupt({
        "prompt": ai_msg,
        "stage": state["stage"],
    })

    # store the filter text for next step (generic search later)
    state["specialist_raw_filter_input"] = str(requested).strip()
    state["csr_query"] = ""
    state["stage"] = "SPECIALIST_FILTERS_CAPTURED"
    return state


##########################################################

    state["specialist_service_specialty"] = str(requested).strip()
    state["csr_query"] = ""
    state["stage"] = "SPECIALIST_SERVICE_CAPTURED"
    return state

####################################################################333

builder.add_node("specialist_ask_filters", node_specialist_ask_filters)

builder.add_edge("specialist_ask_service", "specialist_ask_filters")
builder.add_edge("specialist_ask_filters", END)

###########################################################################3

            elif stage_from_node == "ASK_SPECIALIST_FILTERS":
                ai_resp_text = prompt or last_state.get("ai_response", "")
                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage="ASK_SPECIALIST_FILTERS",
                        ai_response=ai_resp_text,
                        csr_query=req.message or "",
                        prompts=[],
                        prompt_title="",
                        ai_response_code=103,
                        ai_response_type="Dialog",
                    )
                )

