# If the user already typed provider details in the very first free-form text,
# we can skip asking yes/no and go straight to search.
if state.get("knows_provider") is None and not state.get("raw_provider_input"):
    candidate = (state.get("initial_assign_text") or "").strip()
    if candidate:
        parsed = llm_parse_provider_input(candidate)
        if parsed.get("search_type") in ("id", "name_city_state", "zip_only"):
            state["knows_provider"] = True
            state["raw_provider_input"] = candidate
            state["csr_query"] = candidate
            return state



#######################################################

from datetime import datetime, timedelta

def mmddyyyy(dt: datetime) -> str:
    return dt.strftime("%m%d%Y")

def mmddyyyy_slash(dt: datetime) -> str:
    return dt.strftime("%m/%d/%Y")

def node_update_pcp(state: PCPState) -> PCPState:
    """
    Executes PCP update:
      1) terminate current PCP via SOAP (term_dt=today, eff_dt=active effective date)
      2) add new PCP via SOAP (eff_dt=tomorrow, term_dt empty)
      3) verify via get_active_pcp
      4) return confirmation in PromptTitle (AIResponse empty)
    """
    if not (make_change_family_broker_request and build_executeex_envelope and call_execute_ex and extract_short_error):
        state["ai_response"] = "SOAP services are not configured."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        state["last_followup_action"] = "error"
        return state

    new_pid = str(state.get("last_selected_provider_id") or "").strip()
    if not new_pid:
        state["ai_response"] = "Unable to identify the provider to assign. Please select a Provider ID from the list."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 101
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "PROVIDER_FOLLOWUP_RESPONSE"
        state["last_followup_action"] = "other"
        return state

    meme_ck = (state.get("meme_ck") or "").strip()
    grgr_ck = (state.get("grgr_ck") or "").strip()
    curr_pid = (state.get("active_provider_id") or "").strip()
    trsn = (state.get("termination_reason") or "").strip()

    if not (meme_ck and grgr_ck and curr_pid and trsn):
        state["ai_response"] = "Missing required member/termination details to update PCP."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        state["last_followup_action"] = "error"
        return state

    # Dates
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    term_dt = mmddyyyy(today)
    add_eff_dt = mmddyyyy(tomorrow)

    # Termination effective date must come from active PCP
    active_eff = (state.get("active_eff_dt") or "").strip()

    # Config values (not hardcoded)
    pcp_type = getattr(settings, "PCP_TYPE", None) or getattr(settings, "pcp_type", None)
    mctr_orsn = getattr(settings, "MCTR_ORSN", None) or getattr(settings, "mctr_orsn", None)

    if not pcp_type or not mctr_orsn:
        state["ai_response"] = "PCP_TYPE / MCTR_ORSN missing from settings."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        state["last_followup_action"] = "error"
        return state

    try:
        # 1) TERMINATE CURRENT PCP
        term_req = make_change_family_broker_request(
            grgr_ck=grgr_ck,
            meme_ck=meme_ck,
            pcp_type=pcp_type,
            prpr_id=curr_pid,
            eff_dt=active_eff,          # from active PCP
            term_dt=term_dt,            # today
            mctr_trsn=trsn,             # stored termination reason
            mctr_orsn=mctr_orsn,         # from config
        )
        term_env = build_executeex_envelope(term_req)
        term_resp = call_execute_ex(term_env)
        err = extract_short_error(term_resp)
        if err:
            raise RuntimeError(f"Termination failed: {err}")

        # 2) ADD NEW PCP
        add_req = make_change_family_broker_request(
            grgr_ck=grgr_ck,
            meme_ck=meme_ck,
            pcp_type=pcp_type,
            prpr_id=new_pid,
            eff_dt=add_eff_dt,          # tomorrow
            term_dt="",                 # empty
            mctr_trsn="",               # empty
            mctr_orsn="",               # empty
        )
        add_env = build_executeex_envelope(add_req)
        add_resp = call_execute_ex(add_env)
        err = extract_short_error(add_resp)
        if err:
            raise RuntimeError(f"Add PCP failed: {err}")

        # 3) VERIFY
        verify = get_active_pcp(member_key=meme_ck, grgr_ck=grgr_ck)
        active = (verify.get("active") or {}) if isinstance(verify, dict) else {}
        verified_pid = (
            active.get("provId")
            or active.get("providerId")
            or active.get("PCPProviderId")
        )
        verified_pid = str(verified_pid).strip() if verified_pid else ""

        if verified_pid != new_pid:
            raise RuntimeError("PCP update could not be verified. Please try again.")

        # 4) Build confirmation message in PromptTitle (bold values)
        snap = state.get("selected_provider_snapshot") or {}
        name = snap.get("name") or snap.get("providerName") or f"{new_pid}"
        phone = snap.get("phone") or ""
        # full address from your existing formatter
        addr = _format_address_from_provider(snap) if snap else ""

        # Display effective date as MM/DD/YYYY
        eff_disp = mmddyyyy_slash(tomorrow)

        msg = (
            "New PCP has assigned:\n\n"
            f"Name: **{name}**.\n"
            f"Address: **{addr}**.\n"
            f"Phome Number: **{phone}**.\n"
            f"Effective Date: **{eff_disp}**.\n\n"
            "Please allow upto 7 calendar days to receive your new card.\n\n"
            "Do you need further assistance?"
        )

        state["ai_response"] = ""
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 101
        state["prompt_title"] = msg
        state["prompts"] = ["Appointment Scheduling", "Confirmation Fax"]
        state["stage"] = "COMPLETED"
        state["last_followup_action"] = "assign_pcp"
        return state

    except Exception as ex:
        state["ai_response"] = f"An error occurred while updating PCP: {ex}"
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        state["last_followup_action"] = "error"
        return state



##################################################################

if action == "assign_pcp" and chosen:
    pid_final = str(chosen.get("providerId") or pid or "").strip()

    state["last_selected_provider_id"] = pid_final
    state["selected_provider_snapshot"] = chosen
    state["last_followup_action"] = "assign_pcp"
    state["stage"] = "READY_TO_UPDATE_PCP"

    # No response here; next node will do SOAP + confirmation
    state["ai_response"] = ""
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = ""
    state["prompts"] = []
    return state


########################################################################3

