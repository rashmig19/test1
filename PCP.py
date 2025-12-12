def llm_parse_zip_and_date(user_text: str) -> Dict[str, Any]:
    """
    Ask Horizon to extract optional ZIP code and as-of date from user's provider input text.

    Returns JSON:
    {
      "zip": string | null,        # 5-digit ZIP if present
      "as_of_date": string | null  # YYYYMMDD if present, else null
    }
    """
    system = (
        "You are an AI that extracts ZIP code and an as-of date from a member's free-text provider search request.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "zip": string | null,\n'
        '  "as_of_date": string | null\n'
        "}\n"
        "Rules:\n"
        "- zip must be exactly 5 numeric digits if present.\n"
        "- as_of_date must be in YYYYMMDD format if present. If the user mentions a date in another format, "
        "  convert it to YYYYMMDD. If no date is mentioned, use null.\n"
    )
    raw = call_horizon(system, user_text)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        data = json.loads(raw)
        zip_val = data.get("zip")
        as_of = data.get("as_of_date")
        return {
            "zip": zip_val,
            "as_of_date": as_of,
        }
    except Exception:
        logger.warning("Failed to parse ZIP/date JSON from LLM: %s", raw)
        return {"zip": None, "as_of_date": None}




#######################################

if state.get("knows_provider"):
    parsed = llm_parse_provider_input(state.get("raw_provider_input") or "")
    search_type = parsed.get("search_type")

    if search_type == "name_city_state":
        name = parsed.get("name") or ""
        city = parsed.get("city") or ""
        st = parsed.get("state") or ""
        state["provider_name"] = name
        state["provider_city"] = city
        state["provider_state"] = st
        providers = run_provider_search_by_name(name, city, st)
    else:
        # provider ID path...


########################################################


    if search_type == "name_city_state":
        raw_text = state.get("raw_provider_input") or ""

        # Basic name/city/state from existing LLM parser
        name = parsed.get("name") or ""
        city = parsed.get("city") or ""
        st = parsed.get("state") or ""

        state["provider_name"] = name
        state["provider_city"] = city
        state["provider_state"] = st

        # Let Horizon extract optional ZIP and as-of date
        extra = llm_parse_zip_and_date(raw_text)
        zip_code = extra.get("zip")
        as_of_date = extra.get("as_of_date")

        state["startingLocationZip"] = zip_code
        state["asOfDate"] = as_of_date

        # Derive lastName and firstName from provider name
        last_name = ""
        first_name = None
        if name:
            parts = name.split()
            if len(parts) >= 1:
                last_name = parts[0]
            if len(parts) >= 2:
                first_name = parts[1]

        # Use full name for providerName param (if your API uses it)
        full_provider_name = name or None

        member_id = state.get("member_id", "")

        # Call normalized name-search service
        providers = provider_search_by_name(
            last_name=last_name or "",
            group_id=state.get("group_id", ""),
            subscriber_id=state.get("subscriber_id", ""),
            member_id=member_id,
            first_name=first_name,
            provider_name=full_provider_name,
            city=city or None,
            state=st or None,
            startingLocationZip=zip_code,
            asOfDate=as_of_date,
        )

        # Build grid JSON exactly like the ID case
        grid_list: List[Dict[str, Any]] = []
        for p in providers:
            prov_id = (
                p.get("providerId")
                or p.get("provId")
                or p.get("ProviderID")
            )
            name_val = (
                p.get("name")
                or p.get("providerName")
                or p.get("fullName")
                or ""
            )

            addr = _format_address_from_provider(p)

            # Network
            raw_network = (
                p.get("networkStatus")
                or p.get("network")
                or p.get("Network")
            )
            net_str = ""
            if raw_network:
                raw_up = str(raw_network).upper()
                if raw_up in ("IN", "IN NETWORK"):
                    net_str = "In Network"
                elif raw_up in ("OUT", "OUT NETWORK"):
                    net_str = "Out Network"
                else:
                    net_str = str(raw_network)

            # IsAcceptingNewMembers
            accept_val = (
                p.get("isAcceptingNewMembers")
                or p.get("acceptingNewPatients")
                or p.get("AcceptingNewMembers")
            )
            is_accepting = "Y" if str(accept_val).upper() in ("Y", "YES", "TRUE", "1") else "N"

            # PCP assignment indicator
            pcp_val = p.get("pcpAssnInd") or p.get("PCPAssnInd")
            pcp_ind = "Y" if str(pcp_val).upper() in ("Y", "YES", "TRUE", "1") else "N"

            # Distance
            dist = (
                p.get("distance_mi")
                or p.get("distanceInMiles")
                or p.get("Distance_mi")
            )

            grid_list.append({
                "ProviderID": str(prov_id) if prov_id is not None else "",
                "Name": str(name_val),
                "Address": addr,
                "Network": net_str,
                "IsAcceptingNewMembers": is_accepting,
                "PCPAssnInd": pcp_ind,
                "DistanceInMiles": dist,
            })

        response_payload = {"providers": grid_list}

        state["providers_result"] = providers
        state["ai_response"] = json.dumps(response_payload, indent=2, default=str)
        state["prompt_title"] = "Select a provider id to update"
        state["prompts"] = []
        state["ai_response_code"] = 107
        state["ai_response_type"] = "AURA"
        state["stage"] = "SHOW_PROVIDER_LIST"

        requested = interrupt({
            "prompt": state["ai_response"],
            "stage": state["stage"],
        })

        state["csr_query"] = str(requested)
        return state

