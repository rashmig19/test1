class PCPState(TypedDict, total=False):
    # ...
    menu_intent: Optional[str]   # "assign_pcp" | "specialist" | "other"
    flow: Optional[str]          # "pcp" | "specialist"

    service_specialty: Optional[str]  # specialist serviceSpeciality code
    specialist_filters_raw: Optional[str]

########################################################################

def llm_classify_menu_intent(user_text: str) -> str:
    system = (
        "Classify the user's intent for a health plan CSR menu.\n"
        "Return ONLY JSON: {\"intent\": \"assign_pcp\"|\"specialist\"|\"other\"}\n"
        "Rules:\n"
        "- 'assign_pcp' if user asks to assign/change PCP.\n"
        "- 'specialist' if user asks to search/find specialist/provider for a service.\n"
        "- 'other' otherwise.\n"
        "Be robust to spelling mistakes (e.g., 'specalist', 'speclist')."
    )
    raw = call_horizon(system, user_text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        data = json.loads(raw)
        return (data.get("intent") or "other").strip()
    except Exception:
        # fallback (safe)
        t = (user_text or "").lower()
        if "special" in t:
            return "specialist"
        if "pcp" in t:
            return "assign_pcp"
        return "other"


########################

def llm_parse_specialist_filters(user_text: str) -> Dict[str, Any]:
    system = (
        "Extract specialist search filters from free text.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "proceed_default": boolean,\n'
        '  "language": string|null,\n'
        '  "gender": "M"|"F"|null,\n'
        '  "radius_in_miles": number|null,\n'
        '  "zip": string|null,\n'
        '  "as_of_date": string|null\n'
        "}\n"
        "Rules:\n"
        "- proceed_default=true if user says 'proceed with default values' or similar.\n"
        "- zip must be exactly 5 digits if present.\n"
        "- as_of_date must be YYYYMMDD if present; convert common formats.\n"
        "- gender must be M or F only.\n"
        "- language: if user says Spanish -> return \"SPA\". English -> return null.\n"
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
        # Normalize Spanish/English defensively
        lang = data.get("language")
        if isinstance(lang, str):
            lang_up = lang.strip().upper()
            if lang_up in ("SPANISH", "SPA"):
                data["language"] = "SPA"
            elif lang_up in ("ENGLISH", "ENG"):
                data["language"] = None
        # Normalize gender
        g = data.get("gender")
        if isinstance(g, str):
            g2 = g.strip().upper()
            if g2.startswith("F"):
                data["gender"] = "F"
            elif g2.startswith("M"):
                data["gender"] = "M"
            else:
                data["gender"] = None
        # Normalize zip
        z = data.get("zip")
        if isinstance(z, str):
            z = z.strip()
            data["zip"] = z if (len(z) == 5 and z.isdigit()) else None
        return data
    except Exception:
        return {
            "proceed_default": False,
            "language": None,
            "gender": None,
            "radius_in_miles": None,
            "zip": None,
            "as_of_date": None,
        }


######################################

user_choice = (state["csr_query"] or "").strip()
if user_choice in DEFAULT_PROMPTS:
    state["initial_assign_text"] = ""
else:
    state["initial_assign_text"] = user_choice

# NEW: classify intent via Horizon
state["menu_intent"] = llm_classify_menu_intent(user_choice)
if state["menu_intent"] == "specialist":
    state["flow"] = "specialist"
elif state["menu_intent"] == "assign_pcp":
    state["flow"] = "pcp"
else:
    state["flow"] = "other"

return state

###############################################

def node_specialist_ask_service(state: PCPState) -> PCPState:
    # Horizon must produce the sentence
    prompt = call_horizon(
        "You are a CSR assistant. Produce a short HTML paragraph asking the member what services they are looking for at this location. Return only HTML.",
        "Ask: What services are you looking for at this location?"
    ).strip()

    state["ai_response"] = prompt
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 109
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "ASK_SPECIALIST_SERVICE"

    requested = interrupt({
        "prompt": state["ai_response"],
        "stage": state["stage"],
        "prompts": [],
    })
    state["csr_query"] = str(requested).strip()
    return state

###############################

def node_store_specialty(state: PCPState) -> PCPState:
    code = (state.get("csr_query") or "").strip()
    state["service_specialty"] = code
    state["csr_query"] = ""
    return state
#####################

def node_specialist_filters_prompt(state: PCPState) -> PCPState:
    group_id = (state.get("group_id") or "").strip()
    if not group_id:
        state["ai_response"] = "Member details are missing (groupId). Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    default_dist = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")

    html = (
        "<p><b>Ask the below questions:</b></p>\n"
        "<ul>\n"
        "<li>May I know any other preferred Language, if not English and Gender (Male/Female/Either)?</li>\n"
        f"<li>The default distance is {default_dist} miles, please confirm if any changes needed.</li>\n"
        "<li>Search would be performed based on member's home address. If you would like to search provider at different location,please provide with zip code and address line 1.</li>\n"
        "<li>Search will be performed with today’s date unless a different date is provided (MM-DD-YYYY).</li>\n"
        "</ul>"
    )

    state["ai_response"] = html
    state["ai_response_type"] = "Dialog"
    state["ai_response_code"] = 103
    state["prompt_title"] = ""
    state["prompts"] = []  # as per your requirement
    state["stage"] = "ASK_SPECIALIST_FILTERS"

    requested = interrupt({
        "prompt": state["ai_response"],
        "stage": state["stage"],
        "prompts": [],
    })
    state["specialist_filters_raw"] = str(requested).strip()
    return state
##############################################################

def node_run_specialist_search(state: PCPState) -> PCPState:
    group_id = (state.get("group_id") or "").strip()
    subscriber_id = (state.get("subscriber_id") or "").strip()

    if not (group_id and subscriber_id):
        state["ai_response"] = "Member details are missing (groupId/subscriberId). Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    svc = (state.get("service_specialty") or "").strip()
    if not svc:
        state["ai_response"] = "Service specialty is missing. Please provide a service specialty code."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    raw = (state.get("specialist_filters_raw") or "").strip()
    parsed = llm_parse_specialist_filters(raw)

    # defaults
    default_dist = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")

    radius = parsed.get("radius_in_miles")
    if not isinstance(radius, (int, float)):
        radius = default_dist
    radius = int(radius)

    lang = parsed.get("language")  # "SPA" or None (English)
    gender = parsed.get("gender")  # "M"/"F"/None
    zip_code = parsed.get("zip")
    as_of = parsed.get("as_of_date")  # YYYYMMDD or None

    # If no date -> today's YYYYMMDD
    if not as_of:
        as_of = datetime.now().strftime("%Y%m%d")

    # call generic search (you must extend provider_generic_search to accept these params)
    providers = provider_generic_search(
        group_id=group_id,
        subscriber_id=subscriber_id,
        radius_in_miles=radius,
        startingLocationZip=zip_code,
        asOfDate=as_of,
        providerLanguage=lang,
        providerSex=gender,
        serviceSpeciality=svc,
    )

    # build grid (same structure you already use)
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

    state["providers_result"] = providers
    state["ai_response"] = json.dumps({"providers": grid_list}, default=str, separators=(",", ":"))
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 107
    state["prompt_title"] = "Select a provider id to update"  # keep stable
    state["prompts"] = []
    state["stage"] = "SHOW_PROVIDER_LIST"

    requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
    state["csr_query"] = str(requested)
    return state

###########################################

def node_specialist_done_prompt(state: PCPState) -> PCPState:
    state["ai_response"] = ""
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 110
    state["prompt_title"] = "Do you need further assistance?"
    state["prompts"] = ["Yes", "No"]
    state["stage"] = "SPECIALIST_DONE"

    requested = interrupt({
        "prompt": "",
        "stage": state["stage"],
        "prompts": state["prompts"],
    })
    state["csr_query"] = str(requested).strip()
    return state

##############################################33

def node_specialist_close_or_restart(state: PCPState) -> PCPState:
    ans = (state.get("csr_query") or "").strip().lower()

    if "yes" in ans:
        # reset specialist-specific fields only
        state["providers_result"] = None
        state["csr_query"] = ""
        state["service_specialty"] = None
        state["specialist_filters_raw"] = None
        state["last_followup_action"] = None
        state["flow"] = None
        state["menu_intent"] = None
        # go back to start node (menu)
        return state

    # No
    state["ai_response"] = "We're closing your request-feel free to return if you need anything else."
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "COMPLETED"
    return state

###########################

# NEW: Specialist flow does not allow assignment
if state.get("flow") == "specialist" and action == "assign_pcp":
    action = "other"

######################################

def route_from_menu(state: PCPState) -> str:
    intent = (state.get("menu_intent") or "").strip()
    if intent == "assign_pcp":
        return "assign_pcp"
    if intent == "specialist":
        return "specialist"
    return "unsupported"

##########################

def route_after_provider_followup(state: PCPState) -> str:
    # PCP flow
    if state.get("flow") == "pcp":
        if state.get("last_followup_action") == "assign_pcp":
            return "assign"
        return "loop"

    # Specialist flow: after showing address -> completion question
    if state.get("flow") == "specialist":
        if state.get("last_followup_action") == "address":
            return "specialist_done"
        return "loop"

    return "loop"
#################################

builder.add_node("specialist_ask_service", node_specialist_ask_service)
builder.add_node("store_specialty", node_store_specialty)
builder.add_node("specialist_filters_prompt", node_specialist_filters_prompt)
builder.add_node("run_specialist_search", node_run_specialist_search)
builder.add_node("specialist_done_prompt", node_specialist_done_prompt)
builder.add_node("specialist_close_or_restart", node_specialist_close_or_restart)

##############################
builder.add_conditional_edges(
    "start",
    path=route_from_menu,
    path_map={
        "assign_pcp": "assign_pcp_ask_termination",
        "specialist": "specialist_ask_service",
        "unsupported": END,
    },
)

###########################
builder.add_edge("specialist_ask_service", "store_specialty")
builder.add_edge("store_specialty", "specialist_filters_prompt")
builder.add_edge("specialist_filters_prompt", "run_specialist_search")
builder.add_edge("run_specialist_search", "provider_interaction")

###########################
builder.add_conditional_edges(
    "provider_interaction",
    path=route_after_provider_followup,
    path_map={
        "loop": "wait_next_followup",
        "assign": "update_pcp",
        "specialist_done": "specialist_done_prompt",
    }
)


####################
builder.add_edge("specialist_done_prompt", "specialist_close_or_restart")

# If Yes -> go back to start, if No -> END
builder.add_conditional_edges(
    "specialist_close_or_restart",
    path=lambda s: "restart" if "yes" in (s.get("csr_query") or "").lower() else "end",
    path_map={
        "restart": "start",
        "end": END,
    }
)

###########################

elif stage_from_node == "ASK_SPECIALIST_SERVICE":
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="ASK_SPECIALIST_SERVICE",
            ai_response=prompt,
            csr_query=req.message or "",
            prompts=[],
            prompt_title="",
            ai_response_code=109,
            ai_response_type="AURA",
        )
    )

##########################
elif stage_from_node == "ASK_SPECIALIST_FILTERS":
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="ASK_SPECIALIST_FILTERS",
            ai_response=prompt,
            csr_query=req.message or "",
            prompts=[],
            prompt_title="",
            ai_response_code=103,
            ai_response_type="Dialog",
        )
    )

#####################
elif stage_from_node == "SPECIALIST_DONE":
    prompts_list = interrupt_payload.get("prompts") or last_state.get("prompts") or ["Yes", "No"]
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage="SPECIALIST_DONE",
            ai_response="",
            csr_query=req.message or "",
            prompts=prompts_list,
            prompt_title="Do you need further assistance?",
            ai_response_code=110,
            ai_response_type="AURA",
        )
    )

##################3
def provider_generic_search(
    group_id: str,
    subscriber_id: str,
    radius_in_miles: int,
    startingLocationZip: Optional[str] = None,
    asOfDate: Optional[str] = None,
    providerLanguage: Optional[str] = None,
    providerSex: Optional[str] = None,
    serviceSpeciality: Optional[str] = None,
    # keep others optional if you want
) -> List[Dict[str, Any]]:
    # build payload with mandatory + include optional only if not None/empty

#####################3
