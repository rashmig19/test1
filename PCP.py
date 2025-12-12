def llm_parse_provider_input(user_text: str) -> Dict[str, Any]:
    system = (
        "You are an AI that extracts provider search parameters from free text.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "search_type": "id" | "name_city_state" | "zip_only",\n'
        '  "provider_id": string | null,\n'
        '  "zip": string | null,\n'
        '  "name": string | null,\n'
        '  "city": string | null,\n'
        '  "state": string | null\n'
        "}\n"
        "Rules:\n"
        "- If the input is ONLY an 8-digit number, it is a provider_id and search_type must be 'id'.\n"
        "- If the input is ONLY a 5-digit number, it is a ZIP code and search_type must be 'zip_only'.\n"
        "- If the user provides a provider name and city/state (optionally with zip), use search_type='name_city_state'.\n"
        "- zip must be exactly 5 digits if present.\n"
        "Return JSON only (no markdown, no explanation)."
    )
    raw = call_horizon(system, user_text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("Failed to parse provider input JSON from LLM: %s", raw)
        # fallback: treat as name search (safer than calling provider_by_id incorrectly)
        return {
            "search_type": "name_city_state",
            "provider_id": None,
            "zip": None,
            "name": user_text.strip() or None,
            "city": None,
            "state": None,
        }
