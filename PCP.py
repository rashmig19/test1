def llm_parse_no_flow_filters(user_text: str) -> Dict[str, Any]:
    """
    Extract NO-flow filter inputs from user text.

    Returns ONLY JSON:
    {
      "use_defaults": boolean,
      "language": string | null,          # e.g. "Spanish"
      "gender": "M" | "F" | null,         # normalize to M/F
      "zip": string | null,               # 5-digit
      "as_of_date": string | null,        # YYYYMMDD
      "radius_in_miles": number | null    # only if user explicitly wants change
    }
    """
    system = (
        "You are an AI that extracts PCP provider-search filters from text.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "use_defaults": boolean,\n'
        '  "language": string | null,\n'
        '  "gender": "M" | "F" | null,\n'
        '  "zip": string | null,\n'
        '  "as_of_date": string | null,\n'
        '  "radius_in_miles": number | null\n'
        "}\n"
        "Rules:\n"
        "- If the user says 'Please proceed with the default values' (or same meaning), set use_defaults=true.\n"
        "- Otherwise use_defaults=false.\n"
        "- zip must be exactly 5 digits if present.\n"
        "- gender must be M or F when user implies male/female; if unclear use null.\n"
        "- If user provides a date in ANY format, convert to YYYYMMDD.\n"
        "- Only set radius_in_miles if the user explicitly asks to change the default distance.\n"
        "Return JSON only."
    )

    raw = call_horizon(system, user_text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]

    try:
        data = json.loads(raw)
        # Defensive defaults
        return {
            "use_defaults": bool(data.get("use_defaults", False)),
            "language": data.get("language"),
            "gender": data.get("gender"),
            "zip": data.get("zip"),
            "as_of_date": data.get("as_of_date"),
            "radius_in_miles": data.get("radius_in_miles"),
        }
    except Exception:
        logger.warning("Failed to parse NO-flow filter JSON from LLM: %s", raw)
        return {
            "use_defaults": False,
            "language": None,
            "gender": None,
            "zip": None,
            "as_of_date": None,
            "radius_in_miles": None,
        }


####################################

def map_language_to_code(lang_text: Optional[str]) -> str:
    """
    Convert user language like 'Spanish' to providerLanguage code.
    - Uses settings.LANGUAGE_CODE_MAP if present (recommended).
    - Otherwise supports Spanish->SPA (per requirement).
    - English or empty -> "" (default).
    """
    if not lang_text:
        return ""

    raw = str(lang_text).strip().lower()
    if not raw:
        return ""

    # Optional: allow external mapping via config (preferred)
    cfg_map = getattr(settings, "LANGUAGE_CODE_MAP", None)
    if isinstance(cfg_map, dict):
        # case-insensitive match
        for k, v in cfg_map.items():
            if str(k).strip().lower() == raw:
                return str(v or "").strip()

    # Minimal required support (per your requirement)
    if raw in ("spanish", "spa", "es", "español", "espanol"):
        return "SPA"

    # default: treat as English (empty)
    return ""

###################################################

def node_collect_no_flow_filters(state: PCPState) -> PCPState:
    """
    NO flow:
    Show the HTML question block with default distance inserted,
    AIResponseType=Dialog, AIResponseCode=103, PromptTitle empty,
    Prompts=["Please proceed with the default values."].
    Then interrupt and capture user's input in raw_filter_input.
    """
    logger.debug("node_collect_no_flow_filters")

    if state.get("raw_filter_input"):
        return state

    group_id = (state.get("group_id") or "").strip()
    if not group_id:
        state["ai_response"] = "Member details are missing (groupId). Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    # default distance from CSV (Market=groupId, PractitionerType=PCP)
    default_distance = get_default_radius_in_miles(group_id=group_id, practitioner_type="PCP")

    html = (
        "<p><b>Ask the below questions:</b></p>"
        "<ul>"
        "<li>May I know any other preferred Language, if not English and Gender (Male/Female/Either)?</li>"
        f"<li>The default distance is {default_distance} miles, please confirm if any changes needed.</li>"
        "<li>Search would be performed based on member's home address. If you would like to search provider at different location,"
        "please provide with zip code and address line 1.</li>"
        "<li>Search will be performed with today’s date unless a different date is provided (MM-DD-YYYY).</li>"
        "</ul>"
    )

    state["ai_response"] = html
    state["prompt_title"] = ""
    state["prompts"] = ["Please proceed with the default values."]
    state["ai_response_code"] = 103
    state["ai_response_type"] = "Dialog"
    state["stage"] = "ASK_NO_FLOW_FILTERS"

    requested = interrupt({
        "prompt": html,
        "prompts": state["prompts"],
        "stage": state["stage"],
    })

    state["raw_filter_input"] = str(requested).strip()
    state["csr_query"] = state["raw_filter_input"]
    return state

######################################################3

builder.add_node("collect_no_flow_filters", node_collect_no_flow_filters)

######################################3

path_map={
    "knows": "collect_provider_input",
    "unknown": "collect_no_flow_filters",
},

################################

builder.add_edge("collect_no_flow_filters", "run_provider_search")

#########################################################################

else:
    # NO flow -> provider_generic_search
    raw_text = (state.get("raw_filter_input") or "").strip()

    group_id = (state.get("group_id") or "").strip()
    subscriber_id = (state.get("subscriber_id") or "").strip()

    if not group_id or not subscriber_id:
        state["ai_response"] = "Member details are missing (groupId/subscriberId). Please start a new conversation."
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 500
        state["ai_response_type"] = "AURA"
        state["stage"] = "ERROR"
        return state

    default_radius = get_default_radius_in_miles(group_id=group_id, practitioner_type="PCP")

    parsed = llm_parse_no_flow_filters(raw_text)

    # Defaults
    radius = default_radius
    provider_lang = ""
    provider_sex = ""
    zip_code = None
    as_of_date = None

    if parsed.get("use_defaults"):
        # keep defaults only
        pass
    else:
        # language
        provider_lang = map_language_to_code(parsed.get("language"))

        # gender
        g = parsed.get("gender")
        if g in ("M", "F"):
            provider_sex = g

        # zip
        z = parsed.get("zip")
        if z and str(z).isdigit() and len(str(z)) == 5:
            zip_code = str(z)

        # date (YYYYMMDD expected from LLM)
        as_of_date = parsed.get("as_of_date")

        # radius override
        r = parsed.get("radius_in_miles")
        if isinstance(r, (int, float)) and r > 0:
            radius = int(r)

    providers = provider_generic_search(
        group_id=group_id,
        subscriber_id=subscriber_id,
        radius_in_miles=radius,
        startingLocationZip=zip_code or "",
        asOfDate=as_of_date,                 # service should apply today's date if None/empty
        providerLanguage=provider_lang,      # must be supported in provider_search.py
        providersex=provider_sex,            # must be supported in provider_search.py
    )

    # Build SAME grid JSON shape as your other flows
    grid_list: List[Dict[str, Any]] = []
    for p in (providers or []):
        prov_id = p.get("providerId") or ""
        name_val = p.get("name") or ""
        addr = _format_address_from_provider(p)

        raw_network = p.get("networkStatus") or ""
        raw_up = str(raw_network).upper()
        if raw_up in ("IN", "IN NETWORK"):
            net_str = "In Network"
        elif raw_up in ("OUT", "OUT NETWORK"):
            net_str = "Out Network"
        else:
            net_str = str(raw_network) if raw_network else ""

        grid_list.append({
            "ProviderID": str(prov_id),
            "Name": str(name_val),
            "Address": addr,
            "Network": net_str,
            "IsAcceptingNewMembers": p.get("isAcceptingNewMembers"),
            "PCPAssnInd": p.get("pcpAssnInd") or p.get("PCPAssnInd"),
            "DistanceInMiles": p.get("distance_mi") or p.get("distanceInMiles"),
        })

    response_payload = {"providers": grid_list}

    state["providers_result"] = providers
    state["ai_response"] = json.dumps(response_payload, default=str, separators=(",", ":"))
    state["prompt_title"] = "Select a provider id to update"
    state["prompts"] = []
    state["ai_response_code"] = 107
    state["ai_response_type"] = "AURA"
    state["stage"] = "SHOW_PROVIDER_LIST"

    requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
    state["csr_query"] = str(requested)
    return state

############################################################################################

elif stage_from_node == "ASK_NO_FLOW_FILTERS":
    ai_resp_text = prompt or last_state.get("ai_response", "")
    prompts_list = last_state.get("prompts") or ["Please proceed with the default values."]

    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="ASK_NO_FLOW_FILTERS",
            ai_response=ai_resp_text,
            csr_query=req.message or "",
            prompts=prompts_list,
            prompt_title="",
            ai_response_code=103,
            ai_response_type="Dialog",
        )
    )

###############################################################
