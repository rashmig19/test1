raw_text = (state.get("raw_specialist_filters") or "").strip()

        # Use Horizon to extract language/gender/radius/zip/date from raw_text
        # You already have llm_parse_filter_input + llm_parse_zip_and_date.
        filt = llm_parse_filter_input(raw_text)
        extra = llm_parse_zip_and_date(raw_text)

        # language mapping: Spanish -> SPA (you asked)
        lang = (filt.get("language") or "").strip()
        if lang:
            lang_norm = lang.lower()
            if "span" in lang_norm:
                provider_lang = "SPA"
            else:
                provider_lang = ""  # keep default unless you add more mappings
        else:
            provider_lang = ""

        # gender normalize: M/F only
        g = filt.get("gender")
        provider_sex = g if g in ("M", "F") else ""

        group_id = (state.get("group_id") or "").strip()
        subscriber_id = (state.get("subscriber_id") or "").strip()

        # default radius if user didn't override
        default_radius = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")
        radius = filt.get("radius_in_miles")
        radius_in_miles = int(radius) if isinstance(radius, (int, float)) else int(default_radius)

        zip_code = extra.get("zip")
        as_of_date = extra.get("as_of_date")

        providers = provider_generic_search(
            group_id=group_id,
            subscriber_id=subscriber_id,
            radius_in_miles=radius_in_miles,
            asOfDate=as_of_date,
            startingLocationZip=(zip_code or ""),
            providerLanguage=provider_lang,
            providerSex=provider_sex,
            serviceSpeciality=svc,
        )

        # Build same grid JSON
        grid_list: List[Dict[str, Any]] = []
        for p in providers:
            grid_list.append({
                "ProviderID": str(p.get("providerId") or ""),
                "Name": str(p.get("name") or ""),
                "Address": _format_address_from_provider(p),
                "Network": "In Network" if str(p.get("networkStatus") or "").upper() in ("IN", "IN NETWORK") else str(p.get("networkStatus") or ""),
                "IsAcceptingNewMembers": p.get("isAcceptingNewMembers"),
                "PCPAssnInd": p.get("pcpAssnInd"),
                "DistanceInMiles": p.get("distance_mi") or p.get("distanceInMiles"),
            })

        state["providers_result"] = providers
        state["ai_response"] = json.dumps({"providers": grid_list}, separators=(",", ":"))
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 107
        state["prompt_title"] = "Select a provider id to update"
        state["prompts"] = []
        state["stage"] = "SHOW_PROVIDER_LIST"

        requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
        state["csr_query"] = str(requested)
        return state
