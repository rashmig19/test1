def node_collect_specialist_filters(state: PCPState) -> PCPState:
    logger.debug("node_collect_specialist_filters")

    # Only for specialist flow
    if state.get("flow") != "specialist":
        return state

    # If already collected, skip
    if state.get("raw_specialist_filters"):
        return state

    group_id = (state.get("group_id") or "").strip()

    # Default distance from CSV for Specialist
    default_distance = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")

    html = (
        "<p><b>Ask the below questions:</b></p>\n"
        "<ul>\n"
        "<li>May I know any other preferred Language, if not English and Gender (Male/Female/Either)?</li>\n"
        f"<li>The default distance is {default_distance} miles, please confirm if any changes needed.</li>\n"
        "<li>Search would be performed based on member's home address. If you would like to search provider at different location,please provide with zip code and address line 1.</li>\n"
        "<li>Search will be performed with today’s date unless a different date is provided (MM-DD-YYYY).</li>\n"
        "</ul>"
    )

    state["ai_response"] = html
    state["ai_response_type"] = "Dialog"
    state["ai_response_code"] = 103
    state["prompt_title"] = ""
    state["prompts"] = []  # specialist flow: prompts empty
    state["stage"] = "ASK_SPECIALIST_FILTERS"

    requested = interrupt({"prompt": html, "stage": state["stage"]})
    state["raw_specialist_filters"] = str(requested).strip()
    state["csr_query"] = state["raw_specialist_filters"]
    return state
