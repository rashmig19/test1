def llm_decide_followup_action(user_text: str, providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decide what user wants after seeing provider grid.
    Must detect:
      - provider_id_only: user sent only provider id (no other words)
      - address: user asks for address of a provider id
      - assign_pcp: user asks to assign provider as PCP
      - other
    Returns ONLY JSON:
      { "action": "provider_id_only"|"address"|"assign_pcp"|"other", "provider_id": string|null }
    """
    system = (
        "You are an assistant that interprets a member's follow-up message after they received a provider list.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "action": "provider_id_only" | "address" | "assign_pcp" | "other",\n'
        '  "provider_id": string | null\n'
        "}\n"
        "Rules:\n"
        "- If the message contains ONLY a provider id (just digits, no other words), action must be 'provider_id_only'.\n"
        "- If the user asks for address/location of a provider, action='address'.\n"
        "- If the user asks to assign a provider as PCP, action='assign_pcp'.\n"
        "- provider_id must match one of the providerId values from the list when possible.\n"
        "Return JSON only."
    )

    providers_brief = [
        {"providerId": str(p.get("providerId") or ""), "name": p.get("name")}
        for p in (providers or [])
    ]

    payload = {"message": user_text, "providers": providers_brief}
    raw = call_horizon(system, json.dumps(payload)).strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]

    try:
        return json.loads(raw)
    except Exception:
        logger.warning("Failed to parse followup action JSON from LLM: %s", raw)
        return {"action": "other", "provider_id": None}


######################################################

    # If user sent ONLY provider id (no text)
    if action == "provider_id_only":
        state["ai_response"] = "Please check the response that you have provided."
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 101
        state["ai_response_type"] = "AURA"
        state["stage"] = "PROVIDER_ACTION"
        state["csr_query"] = ""
        return state


##########################################################33

    if action == "address" and chosen:
        prov_id = str(chosen.get("providerId") or "")
        addr_payload = {
            "providerAddress": [{
                "ProviderID": prov_id,
                "AddressType": chosen.get("addressType"),
                "AddressLine1": chosen.get("address1"),
                "AddressLine2": chosen.get("address2"),
                "City": chosen.get("city"),
                "State": chosen.get("state"),
                "Zip": chosen.get("zip"),
                "Country": chosen.get("country") or chosen.get("county"),
                "Phone": chosen.get("phone"),
                "Fax": chosen.get("fax"),
                "EffectiveDate": chosen.get("effectiveDate"),
                "TerminationDate": chosen.get("terminationDate"),
            }]
        }

        state["ai_response"] = json.dumps(addr_payload, indent=2, default=str)
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 108
        state["ai_response_type"] = "AURA"
        state["stage"] = "SHOW_PROVIDER_ADDRESS_JSON"
        state["csr_query"] = ""
        return state
