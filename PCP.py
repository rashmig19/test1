def llm_validate_no_px_keys(filtered_case: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You validate JSON objects.\n"
        "Check whether any key anywhere in the JSON starts with 'px' (case-insensitive).\n"
        "Return ONLY JSON:\n"
        "{\"has_px\": boolean, \"examples\": [\"pxKey1\", \"pxKey2\"]}\n"
        "examples should include up to 5 offending keys if any."
    )
    raw = call_horizon(system, json.dumps(filtered_case)[:12000])  # cap to avoid token blowups
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        i = raw.find("{")
        if i != -1:
            raw = raw[i:]
    try:
        return json.loads(raw)
    except Exception:
        return {"has_px": None, "examples": []}
