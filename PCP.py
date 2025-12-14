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
        "- If there is a 5-digit ZIP and no provider id and no name/city/state, use search_type='zip_only' and set zip.\n"
        "- If there is a provider name with city/state (and optional zip), use search_type='name_city_state'.\n"
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


##################################################################

def node_collect_knows_provider(state: PCPState) -> PCPState:
    logger.debug("node_collect_knows_provider")

    # ✅ FREE-FORM SHORTCIRCUIT:
    # If user already provided provider info in the initial menu message
    # (e.g., "Please assign 12345678 as PCP" or "Please assign John, Dallas, TX as PCP"),
    # skip ASK_KNOWS_PROVIDER and go straight to provider search.
    if state.get("knows_provider") is None and not state.get("raw_provider_input"):
        candidate = (state.get("initial_assign_text") or "").strip()
        if candidate:
            parsed = llm_extract_provider_query_from_assign_text(candidate)

            st = parsed.get("search_type")
            pid = (parsed.get("provider_id") or "").strip() if parsed.get("provider_id") else ""
            z = (parsed.get("zip") or "").strip() if parsed.get("zip") else ""
            name = (parsed.get("name") or "").strip() if parsed.get("name") else ""
            city = (parsed.get("city") or "").strip() if parsed.get("city") else ""
            st_code = (parsed.get("state") or "").strip() if parsed.get("state") else ""

            if st == "id" and pid:
                state["knows_provider"] = True
                state["raw_provider_input"] = pid
                state["csr_query"] = pid
                state["initial_assign_text"] = ""  # consume it
                return state

            if st == "zip_only" and z:
                state["knows_provider"] = True
                state["raw_provider_input"] = z
                state["csr_query"] = z
                state["initial_assign_text"] = ""
                return state

            if st == "name_city_state" and (name or city or st_code or z):
                # Build a normalized single string input for downstream LLM parser
                parts = [p for p in [name, city, st_code, z] if p]
                normalized = ", ".join(parts) if parts else candidate

                state["knows_provider"] = True
                state["raw_provider_input"] = normalized
                state["csr_query"] = normalized
                state["initial_assign_text"] = ""
                return state

    # ---- existing behavior unchanged ----
    if state.get("knows_provider") is None:
        system_prompt = (
            "You are a CSR assistant."
            "Write a short, clear question asking the member whether they know"
            "the name or ID of the provider they are looking for."
            "The question should be suitable as a UI title and must be answerable with Yes or No."
            "Return ONLY the question text, without any extra words."
        )
        user_prompt = "Ask the member if they know the name or ID of the provider they are looking for."
        ai_msg = call_horizon(system_prompt, user_prompt).strip()
        state["ai_response"] = ""
        state["prompt_title"] = ai_msg
        state["prompts"] = ["Yes", "No"]
        state["ai_response_code"] = 101
        state["ai_response_type"] = "AURA"
        state["stage"] = "ASK_KNOWS_PROVIDER"

        requested = interrupt({
            "prompt": ai_msg,
            "prompts": state["prompts"],
            "stage": state["stage"],
        })
        answer = str(requested).strip().lower()
        state["csr_query"] = answer
        state["knows_provider"] = "yes" in answer

        logger.debug("knows_provider captured from interrupt: %s", state["knows_provider"])
        return state
    else:
        return state

#################################################################################################

