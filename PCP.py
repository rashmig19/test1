    specialist_service_specialty: Optional[str]
    specialist_raw_filter_input: Optional[str]
####################################################33

def llm_parse_specialist_filters(user_text: str) -> Dict[str, Any]:
    """
    Extract specialist search filters from text.

    Returns ONLY JSON:
    {
      "use_defaults": boolean,
      "language": string | null,
      "gender": "M" | "F" | null,
      "zip": string | null,
      "as_of_date": string | null,        # YYYYMMDD
      "radius_in_miles": number | null
    }
    """
    system = (
        "You are an AI that extracts Specialist provider-search filters from text.\n"
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
        "- Only set radius_in_miles if the user explicitly wants to change the default distance.\n"
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
        return {
            "use_defaults": bool(data.get("use_defaults", False)),
            "language": data.get("language"),
            "gender": data.get("gender"),
            "zip": data.get("zip"),
            "as_of_date": data.get("as_of_date"),
            "radius_in_miles": data.get("radius_in_miles"),
        }
    except Exception:
        logger.warning("Failed to parse Specialist filter JSON from LLM: %s", raw)
        return {
            "use_defaults": False,
            "language": None,
            "gender": None,
            "zip": None,
            "as_of_date": None,
            "radius_in_miles": None,
        }


###############################################3


def node_run_specialist_generic_search(state: PCPState) -> PCPState:
    """
    Specialist flow:
    - Use stored specialist_service_specialty
    - Parse specialist_raw_filter_input (any combo)
    - Apply defaults (distance from CSV for Specialist + today's YYYYMMDD if date missing)
    - Call provider_generic_search(serviceSpecialty=...)
    - Return provider grid JSON in AIResponse (AURA, 107), PromptTitle/Prompts empty
    """
    logger.debug("node_run_specialist_generic_search")

    # If already have results, don't re-run
    if state.get("providers_result"):
        return state

    group_id = (state.get("group_id") or "").strip()
    subscriber_id = (state.get("subscriber_id") or "").strip()
    specialty = (state.get("specialist_service_specialty") or "").strip()

    if not group_id or not subscriber_id:
        state["ai_response"] = "Member details are missing (groupId/subscriberId). Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    if not specialty:
        state["ai_response"] = "Service specialty code is missing. Please enter the service specialty code."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 109
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ASK_SPECIALTY"
        return state

    # Defaults
    default_radius = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")
    today_yyyymmdd = datetime.now().strftime("%Y%m%d")

    raw_text = (state.get("specialist_raw_filter_input") or "").strip()
    parsed = llm_parse_specialist_filters(raw_text)

    radius = int(default_radius)
    provider_lang = ""
    provider_sex = ""
    zip_code = ""
    as_of_date = today_yyyymmdd

    if parsed.get("use_defaults"):
        # keep defaults only
        pass
    else:
        # language -> providerLanguage
        provider_lang = map_language_to_code(parsed.get("language"))

        # gender -> providerSex (M/F only)
        g = parsed.get("gender")
        if g in ("M", "F"):
            provider_sex = g

        # zip -> startingLocationZip
        z = parsed.get("zip")
        if z and str(z).isdigit() and len(str(z)) == 5:
            zip_code = str(z)

        # date -> asOfDate (YYYYMMDD)
        d = parsed.get("as_of_date")
        if isinstance(d, str) and d.strip():
            as_of_date = d.strip()

        # radius override
        r = parsed.get("radius_in_miles")
        if isinstance(r, (int, float)) and r > 0:
            radius = int(r)

    # ✅ IMPORTANT: provider_generic_search must accept these named params
    providers = provider_generic_search(
        group_id=group_id,
        subscriber_id=subscriber_id,
        radius_in_miles=radius,
        startingLocationZip=zip_code,
        asOfDate=as_of_date,
        providerLanguage=provider_lang,
        providerSex=provider_sex,
        serviceSpecialty=specialty,
    )

    # Build grid JSON (same shape)
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
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 107
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "SHOW_SPECIALIST_PROVIDER_LIST"

    requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
    state["csr_query"] = str(requested)
    return state

####################################################################################################33

builder.add_node("run_specialist_generic_search", node_run_specialist_generic_search)

builder.add_edge("specialist_ask_filters", "run_specialist_generic_search")

elif stage_from_node == "SHOW_SPECIALIST_PROVIDER_LIST":
    ai_resp_text = prompt or last_state.get("ai_response", "")
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="SHOW_SPECIALIST_PROVIDER_LIST",
            ai_response=ai_resp_text,
            csr_query=req.message or "",
            prompts=[],
            prompt_title="",
            ai_response_code=107,
            ai_response_type="AURA",
        )
    )

####################################################################################################
