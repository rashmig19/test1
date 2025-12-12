    if search_type == "zip_only":
        raw_text = state.get("raw_provider_input") or ""

        # ZIP from provider input parser
        zip_code = parsed.get("zip")
        if not zip_code:
            # safety fallback: treat the entire text as zip if LLM didn’t fill it
            zip_code = raw_text.strip()

        # As-of-date from LLM (YYYYMMDD or None)
        extra = llm_parse_zip_and_date(raw_text)
        as_of_date = extra.get("as_of_date")

        state["startingLocationZip"] = zip_code
        state["asOfDate"] = as_of_date

        # Mandatory fields for generic search
        group_id = state.get("group_id", "")
        subscriber_id = state.get("subscriber_id", "")
        if not group_id or not subscriber_id:
            state["ai_response"] = "Member details are missing (groupId/subscriberId). Please start a new conversation."
            state["prompt_title"] = "Error"
            state["prompts"] = []
            state["ai_response_code"] = 500
            state["ai_response_type"] = "Dialog"
            state["stage"] = "ERROR"
            return state

        # Radius from Default Distance.csv (Market=groupId, PractitionerType=PCP)
        try:
            radius = get_default_radius_in_miles(group_id=group_id, practitioner_type="PCP")
        except Exception as ex:
            state["ai_response"] = f"Unable to determine default search radius: {ex}"
            state["prompt_title"] = "Error"
            state["prompts"] = []
            state["ai_response_code"] = 500
            state["ai_response_type"] = "Dialog"
            state["stage"] = "ERROR"
            return state

        # Call generic search (normalized providers list)
        providers = provider_generic_search(
            group_id=group_id,
            subscriber_id=subscriber_id,
            radius_in_miles=radius,
            startingLocationZip=str(zip_code)[:5],
            asOfDate=as_of_date,
        )

        # Build grid JSON exactly like provider-id case
        grid_list: List[Dict[str, Any]] = []
        for p in providers:
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

            accept_val = p.get("isAcceptingNewMembers")
            is_accepting = "Y" if str(accept_val).upper() in ("Y", "YES", "TRUE", "1") else "N"

            pcp_val = p.get("pcpAssnInd")
            pcp_ind = "Y" if str(pcp_val).upper() in ("Y", "YES", "TRUE", "1") else "N"

            dist = p.get("distance_mi") or p.get("distanceInMiles")

            grid_list.append({
                "ProviderID": str(prov_id),
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

        requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
        state["csr_query"] = str(requested)
        return state
