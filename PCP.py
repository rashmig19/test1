def llm_extract_provider_query_from_assign_text(user_text: str) -> Dict[str, Any]:
    """
    Extract provider search query from a free-form 'assign ... as pcp' message.

    Returns ONLY JSON:
    {
      "search_type": "id" | "name_city_state" | "zip_only" | "unknown",
      "provider_id": string | null,
      "zip": string | null,
      "name": string | null,
      "city": string | null,
      "state": string | null
    }
    """
    system = (
        "You extract provider-identifying info from a user request that intends to assign a PCP.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "search_type": "id" | "name_city_state" | "zip_only" | "unknown",\n'
        '  "provider_id": string | null,\n'
        '  "zip": string | null,\n'
        '  "name": string | null,\n'
        '  "city": string | null,\n'
        '  "state": string | null\n'
        "}\n"
        "Rules:\n"
        "- If there is an 8-digit provider id anywhere in the text, use search_type='id' and set provider_id.\n"
        "- Else if there is a provider name anywhere in the text (even if city/state/zip are NOT present), "
        "use search_type='name_city_state' and set name. city/state/zip may be null.\n"
        "- Else if there is a 5-digit ZIP and no provider id and no name, use search_type='zip_only' and set zip.\n"
        "- Ignore filler words like 'please', 'assign', 'as pcp', etc.\n"
        "Return JSON only."
    )

    raw = call_horizon(system, user_text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        i = raw.find("{")
        if i != -1:
            raw = raw[i:]
    try:
        data = json.loads(raw)
        st = (data.get("search_type") or "unknown").strip()
        if st not in ("id", "name_city_state", "zip_only"):
            st = "unknown"
        return {
            "search_type": st,
            "provider_id": data.get("provider_id"),
            "zip": data.get("zip"),
            "name": data.get("name"),
            "city": data.get("city"),
            "state": data.get("state"),
        }
    except Exception:
        logger.warning("Failed to parse assign-text provider JSON from LLM: %s", raw)
        return {
            "search_type": "unknown",
            "provider_id": None,
            "zip": None,
            "name": None,
            "city": None,
            "state": None,
        }


##############################################3

