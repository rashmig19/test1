import json
from typing import Any, Dict, List
from app.graph.state import PCPState
from app.services.case_rest import(
    get_case,
    sync_case_from_interaction
)
from app.llm.horizon_client import call_horizon
from app.llm.parsers import (
    llm_validate_no_px_keys, 
    llm_parse_filter_input,
    llm_extract_provider_query_from_assign_text,
    llm_decide_followup_action,
    llm_parse_provider_input,
    llm_parse_zip_and_date,
    llm_parse_no_flow_filters,
    llm_parse_specialist_filters
)
from app.config import settings
from datetime import datetime, timedelta, date
from langgraph.types import Command, interrupt
from app.services.member_rest import (
    member_search,
    get_active_pcp
)
from app.services.provider_search import(
    get_default_radius_in_miles,
    provider_search_by_name,
    _format_address_from_provider,
    provider_search_by_id,
    provider_generic_search,
    map_language_to_code
)
from app.services.soap_client import(
    make_change_family_broker_request,
    build_executeex_envelope,
    call_execute_ex,
    format_date_mmddyyyy,
    extract_short_error
)
from app.graph.call_source import mark_api, mark_llm
from fastapi import FastAPI, HTTPException, status
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.graph.routing import (
    route_after_provider_followup,
    route_after_specialist_completion,
    route_from_menu,
    route_after_specialist_provider_address,
    route_knows_provider
)

logger = logging.getLogger("pcp_app")

def node_load_case(state: PCPState) -> PCPState:
    """
    Load case details using InteractionID, filter out px* keys, store in state.
    No interrupt here; continue to menu/start.
    """
    # If already loaded (e.g., resume), don't redo
    if state.get("case_filtered") is not None:
        return state

    interaction_id = (state.get("interaction_id") or "").strip()
    if not interaction_id:
        # If UI didn't send it, keep flow alive but record nothing
        state["case_raw"] = None
        state["case_filtered"] = None
        return state

    try:
        mark_api(state, get_case)
        raw, filtered = sync_case_from_interaction(interaction_id)
        # filtered = _strip_px_keys(raw, prefix="px")
        state["case_raw"] = raw
        state["case_filtered"] = filtered

        # LLM validation
        mark_llm(state)
        validated = llm_validate_no_px_keys(filtered)
        state["case_px_audit"] = validated

        return state

    except Exception as ex:
        logger.exception("Get Case API failed: %s", ex)
        # Do NOT break the existing PCP flow; just store empty and continue
        state["case_raw"] = None
        state["case_filtered"] = None
        return state
    
def node_start(state: PCPState) -> PCPState:
    """
    Initial node: show main menu and interrupt to get user's choice.
    """
    logger.debug("node_start")
    menu_text = (
        "You can choose one of the following options:<br>"
        "1. Assign PCP<br>"
        "2. Search for specialist<br>"
        "3. Need something else"
    )
    mark_llm(state)
    ai_msg = call_horizon(
        "You are a helpful CSR assistant. Ask the member to choose one of the options.",
        menu_text,
    )    
    state["ai_response"] = ai_msg
    state["prompt_title"] = "How can I assist you today?"
    state["prompts"] = settings.DEFAULT_PROMPTS
    state["ai_response_code"] = 100
    state["ai_response_type"] = "Dialog"
    state["stage"] = "MENU"

    # Interrupt, waiting for user choice
    requested = interrupt({
        "prompt": ai_msg,
        "prompts": settings.DEFAULT_PROMPTS,
        "stage": state["stage"],
        "call_source": state.get("call_name") or state.get("call_type"),
    })

    # When resumed:
    state["csr_query"] = str(requested)

    # Save original text; later we can reuse if it already includes provider id /name
    user_choice = (state["csr_query"] or "").strip()

    if user_choice in settings.DEFAULT_PROMPTS:
        state["initial_assign_text"] = ""
    else:
        state["initial_assign_text"] = user_choice
    return state


def node_collect_termination_reason(state: PCPState) -> PCPState:
    logger.debug("node_collect_termination_reason")

    # If already present, just return (safety)
    if state.get("termination_reason"):
        return state
    
    # Consume the termination reason that was captured in csr_query
    reason = (state.get("csr_query") or "").strip()
    if reason:
        state["termination_reason"] = reason
        state["csr_query"] = ""
        return state
    
    # Safety fallback: if somehow we reached here without a resume value,
    # route back to asking termination again by leaving termination_reason empty
    return state

def node_assign_pcp_ask_termination(state: PCPState) -> PCPState:
    """
    Node used when user says 'Assign PCP' in chat.
    1) Calls get_active_pcp with member_id from state
    2) Extracts Active_Provider_ID
    3) Asks: 'Please select the termination reason for current PCP - <Active_Provider_ID>.'
    4) AIResponseCode = 112
    5) Interrupts to wait for user's termination reason
    """
    logger = logging.getLogger("pcp.assign_pcp")
    logger.debug("node_assign_pcp_ask_termination, state=%s", state)

    member_id = state.get("member_id")
    if not member_id:
        # Safety guard
        state["ai_response"] = "Member information is missing. Please start a new conversation."
        state["ai_response_code"] = 500
        state["ai_response_type"] = "Dialog"
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    # ----------------------------------------------------------------------
    # If we *don't* have termination_reason yet: fetch active PCP and ask
    # ----------------------------------------------------------------------
    if not state.get("termination_reason"):
        # 1) Call your REST tool: get_active_pcp
        try:
            member_response = member_search(dob = "", mbrId=member_id, firstNm = "", lastNm = "")
            # mark_api(state, member_search) # If we return right after this API call in an error case, this helps
            if isinstance(member_response, list):
                if not member_response:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Member not found for the given mbrId."
                    )
                member_payload = member_response[0]
            else:
                member_payload = member_response
            
            state["grgr_ck"] = str(member_payload.get("grgrCk") or member_payload.get("sbsbCk")).strip()
            state["meme_ck"] = str(member_payload.get("memeCk")).strip()
            state["group_id"] = str(member_payload.get("grpId")).strip()
            state["subscriber_id"] = str(member_payload.get("subscriberId")).strip()

            # print("DEBUG get_active_pcp = ", get_active_pcp, type(get_active_pcp))
            active_pcp = get_active_pcp(member_key=state["meme_ck"], grgr_ck=state["grgr_ck"])
            # print("active_pcp: ", active_pcp)
            # mark_api(state, get_active_pcp) # If we return right after this API call in an error case, this helps
            curr_active = active_pcp.get("active") or {}

            # effective date present in active object
            eff = (
                curr_active.get("effectiveDate")
                or curr_active.get("provEffectiveDt")
                or curr_active.get("effDt")
            )
            state["active_eff_dt"] = str(eff).strip() if eff else None

        except Exception as ex:
            logger.exception("get_active_pcp failed: %s", ex)
            mark_api(state, get_active_pcp) # Here, error step shows which API faield
            state["ai_response"] = "Unable to fetch your current PCP details right now. Please try again later."
            state["ai_response_code"] = 500
            state["ai_response_type"] = "Dialog"
            state["prompts"] = []
            state["stage"] = "ERROR"
            return state

        # 2) Extract Provider ID from response (adjust key as per your API)
        #    example keys: 'provId', 'ProviderId', etc.
        active_provider_id = (
            curr_active.get("provId")
            or curr_active.get("providerId")
            or curr_active.get("PCPProviderId")
        )
        # print("active_provider_id: ", active_provider_id)
        state["active_provider_id"] = str(active_provider_id) if active_provider_id else ""

        # 3) Build question including active provider id
        base_question = f"Please select the termination reason for current PCP - {state['active_provider_id']}."

        # 4) Optional: pass through Horizon LLM to keep your 'LLM brain' requirement
        ai_msg = call_horizon(
            "You are a CSR assistant. Ask the member to select a termination reason using the given sentence. "
            "Keep the provider ID exactly as it is.",
            base_question,
        )
        mark_llm(state)

        # 5) Populate response fields
        state["ai_response"] = ai_msg
        state["ai_response_code"] = 112          # 112 = Dialog
        state["ai_response_type"] = "Dialog"
        state["prompts"] = []                    # UI will show its own reasons list
        state["stage"] = "WAIT_TERMINATION_REASON"

        # 6) INTERRUPT – return this message to the caller (chat endpoint)
        requested = interrupt({
            "prompt": state["ai_response"],
            "stage": state["stage"],
            "active_provider_id": state["active_provider_id"],
        })

        # When resumed (next /chat call), we capture termination reason:
        # state["termination_reason"] = str(requested)
        # logger.debug("Termination reason captured from interrupt: %s", state["termination_reason"])
        
        # On resume, capture the termination reason into csr_query for the next node
        state["csr_query"] = str(requested).strip()
        return state

    # ----------------------------------------------------------------------
    # Else, termination_reason is already filled (we resumed this node),
    # so we just handoff to the next stage in your flow (ask provider id / yes-no etc.)
    # ----------------------------------------------------------------------
    return state

def node_collect_knows_provider(state: PCPState) -> PCPState:
    logger.debug("node_collect_knows_provider")

    # FREE-FORM SHORTCIRCUIT:
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

            # Name only should also skip YES/NO for free form text
            if (st == "name_city_state" or (st == "unknown" and name)) and (name or city or st_code or z):
                # Build a normalized single string input for downstream LLM parser
                # For name only, normalized will just be the name
                parts = [p for p in [name, city, st_code, z] if p]
                normalized = ", ".join(parts) if parts else candidate

                state["knows_provider"] = True
                state["raw_provider_input"] = normalized
                state["csr_query"] = normalized
                state["initial_assign_text"] = ""
                return state
  
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
        mark_llm(state)
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

def node_collect_provider_input(state: PCPState) -> PCPState:
    logger.debug("node_collect_provider_input")

    if not state.get("raw_provider_input"):
        system_prompt = (
            "You are a CSR assistant helping a member choose a new PCP. "
            "You must produce a short dialog message that will be shown in the AIResponse field. "
            "The message MUST:<br>"
            "- Start with a bold line: 'Ask the below questions:'  "
            "- Then show three bullet points (or dotted list items) in this exact order and wording:  "
            " 1) 'May I know the Provider ID? or Provider Name, City and State?'  "
            " 2) 'Search would be performed based on member's home address. If you would like to search provider at different location, please provide with zip code and address line 1.'  "
            " 3) 'Search will be performed with today's date unless a different date is provided (MM-DD-YYYY).'  "
            "Use simple markdown-style formatting: bold for the first line, and each question on its own line starting with a bullet or a dot. "
            "Return ONLY the formatted text, no explanations."
        )
        user_prompt = "Generate the dialog text exactly as specified, to be displayed to the CSR as instructions."
        ai_msg = call_horizon(system_prompt, user_prompt).strip()
        mark_llm(state)
        state["ai_response"] = ai_msg
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 103
        state["ai_response_type"] = "Dialog"
        state["stage"] = "ASK_PROVIDER_INPUT"

        requested = interrupt({
            "prompt": ai_msg,
            "stage": state["stage"],
        })
        state["raw_provider_input"] = str(requested)
        state["csr_query"] = state["raw_provider_input"]
        return state
    else:
        return state

def node_collect_no_flow_filters(state: PCPState) -> PCPState:
    """
    NO flow:
    Show the HTML question block with default distance inserted,
    AIResponseType=Dialog, AIResponseCode=103, PromptTitle empty,
    Prompts=["Please proceed with the default values."].
    Then interrupt and capture user's input in raw_filter_input.
    """
    logger.debug("node_collect_no_flow_filters")
    mark_llm(state) 

    if state.get("raw_filter_input"):
        return state

    group_id = (state.get("group_id") or "").strip()
    if not group_id:
        state["ai_response"] = "Member details are missing (groupId). Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    # default distance from CSV (Market=groupId, PractitionerType=PCP)
    default_distance = get_default_radius_in_miles(group_id=group_id, practitioner_type="PCP")
  
    html = (
        "<p><b>Ask the below questions:</b></p>"
        "<ul>"
        "<li>May I know any other preferred Language, if not English and Gender (Male/Female/Either)?</li>"
        f"<li>The default distance is {default_distance} miles, please confirm if any changes needed.</li>"
        "<li>Search would be performed based on member's home address. If you would like to search provider at different location,"
        "please provide with zip code and address line 1.</li>"
        "<li>Search will be performed with today’s date unless a different date is provided (MM-DD-YYYY).</li>"
        "</ul>"
    )

    state["ai_response"] = html
    state["prompt_title"] = ""
    state["prompts"] = ["Please proceed with the default values."]
    state["ai_response_code"] = 103
    state["ai_response_type"] = "Dialog"
    state["stage"] = "ASK_NO_FLOW_FILTERS"

    requested = interrupt({
        "prompt": html,
        "prompts": state["prompts"],
        "stage": state["stage"],
    })

    state["raw_filter_input"] = str(requested).strip()
    state["csr_query"] = state["raw_filter_input"]
    return state

def node_collect_filter_input(state: PCPState) -> PCPState:
    logger.debug("node_collect_filter_input")
    if not state.get("raw_filter_input"):
        ai_msg = call_horizon(
            "You are a CSR assistant.",
            "Ask the member for any of these PCP search filters: language preference, radius in miles, gender. "
            "They may provide all or any subset.",
        )
        mark_llm(state)
        state["ai_response"] = ai_msg
        state["prompt_title"] = "Search filters"
        state["prompts"] = []
        state["ai_response_code"] = 104
        state["ai_response_type"] = "Dialog"
        state["stage"] = "ASK_FILTERS"

        requested = interrupt({
            "prompt": ai_msg,
            "stage": state["stage"],
        })
        state["raw_filter_input"] = str(requested)
        state["csr_query"] = state["raw_filter_input"]
        return state
    else:
        return state

def node_specialist_ask_service(state: PCPState) -> PCPState:
    """
    Specialist flow step 1:
    Ask exact question (AIResponse must be EXACT HTML),
    then interrupt and store the service specialty code.
    """
    logger.debug("node_specialist_ask_service")

    # mark flow
    state["flow"] = "specialist"

    # Ask Horizon to output EXACT text, then enforce exact output safely
    system = (
        "You are a CSR assistant. You MUST return EXACTLY the following text and nothing else:\n"
        f"{settings.SPECIALIST_SERVICE_QUESTION}"
    )
    ai_msg = call_horizon(system, "Return the exact text now.").strip()
    mark_llm(state)

    # Enforce exactness (requirement says exact string)
    if ai_msg != settings.SPECIALIST_SERVICE_QUESTION:
        ai_msg = settings.SPECIALIST_SERVICE_QUESTION

    state["ai_response"] = ai_msg
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 109
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "ASK_SPECIALIST_SERVICE"

    requested = interrupt({
        "prompt": ai_msg,
        "stage": state["stage"],
    })

    # On resume: store the specialty code for later steps
    state["specialist_service_specialty"] = str(requested).strip()
    state["csr_query"] = ""  # keep clean
    state["stage"] = "SPECIALIST_SERVICE_CAPTURED"

    return state

def node_specialist_ask_filters(state: PCPState) -> PCPState:
    """
    After specialist service specialty code is captured:
    - compute default distance from CSV using PractitionerType="Specialist" and group_id
    - ask Horizon to output the exact HTML template with default_distance replaced
    - AIResponseType=Dialog, AIResponseCode=103, PromptTitle/Prompts empty
    - interrupt to capture user filter inputs for next step (later)
    """
    logger.debug("node_specialist_ask_filters")

    # Ensure we have member market identifiers (group_id/subscriber_id) for CSV + later API calls
    member_id = (state.get("member_id") or "").strip()
    if not member_id:
        state["ai_response"] = "Member information is missing. Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    if not (state.get("group_id") and state.get("subscriber_id") and state.get("meme_ck") and state.get("grgr_ck")):
        try:
            member_response = member_search(dob="", mbrId=member_id, firstNm="", lastNm="")
            if isinstance(member_response, list):
                if not member_response:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
                member_payload = member_response[0]
            else:
                member_payload = member_response

            state["grgr_ck"] = str(member_payload.get("grgrCk") or member_payload.get("sbsbCk") or "").strip()
            state["meme_ck"] = str(member_payload.get("memeCk") or "").strip()
            state["group_id"] = str(member_payload.get("grpId") or "").strip()
            state["subscriber_id"] = str(member_payload.get("subscriberId") or "").strip()
        except Exception as ex:
            logger.exception("member_search failed in specialist flow: %s", ex)
            mark_api(state, member_search)
            state["ai_response"] = "Unable to fetch member details right now. Please try again later."
            state["ai_response_type"] = "AURA"
            state["ai_response_code"] = 500
            state["prompt_title"] = ""
            state["prompts"] = []
            state["stage"] = "ERROR"            
            return state

    group_id = (state.get("group_id") or "").strip()
    if not group_id:
        state["ai_response"] = "Member group information is missing. Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    # Default distance from CSV for Specialist
    try:
        default_distance = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")
    except Exception as ex:
        logger.exception("Default distance lookup failed: %s", ex)
        state["ai_response"] = "Unable to determine the default distance right now. Please try again later."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    expected_html = settings.SPECIALIST_FILTERS_TEMPLATE.replace("<<default_distance>>", str(default_distance))

    # Ask Horizon to produce EXACT HTML (then enforce exactness to meet UI contract)
    system = (
        "You are a CSR assistant.\n"
        "Return EXACTLY the following HTML (no extra spaces, no explanation, no markdown fences):\n"
        f"{expected_html}"
    )
    ai_msg = call_horizon(system, "Return the exact HTML now.").strip()
    mark_llm(state)
    if ai_msg != expected_html:
        ai_msg = expected_html  # enforce contract

    state["ai_response"] = ai_msg
    state["ai_response_type"] = "Dialog"
    state["ai_response_code"] = 103
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "ASK_SPECIALIST_FILTERS"

    requested = interrupt({
        "prompt": ai_msg,
        "stage": state["stage"],
    })

    # store the filter text for next step (generic search later)
    state["specialist_raw_filter_input"] = str(requested).strip()
    state["csr_query"] = ""
    state["stage"] = "SPECIALIST_FILTERS_CAPTURED"
    return state

def node_specialist_provider_address(state: PCPState) -> PCPState:
    """
    After specialist provider grid is shown, user selects a provider id and asks for address.
    Return address JSON in AIResponse, and mark conversation as completed (inquiry only).
    """
    logger.debug("node_specialist_provider_address")

    providers = state.get("providers_result") or []
    if not providers:
        state["ai_response"] = "No providers are available in the current result. Please start a new search."
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 404
        state["ai_response_type"] = "AURA"
        state["stage"] = "ERROR"
        return state

    # We should already have the user's message in csr_query (resume value)
    user_msg = (state.get("csr_query") or "").strip()
    if not user_msg:
        requested = interrupt({"prompt": "", "stage": "WAIT_SPECIALIST_PROVIDER_ACTION"})
        state["csr_query"] = str(requested).strip()
        user_msg = state["csr_query"]

    # Use existing LLM helper to understand intent + provider id
    decision = llm_decide_followup_action(user_msg, providers)
    mark_llm(state)
    action = decision.get("action")
    print("decision : ", decision, "action: ", action)
    pid = decision.get("provider_id")

    if action == "provider_id_only":
        mark_llm(state)
        state["ai_response"] = "Please check the response that you have provided"
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 101
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "SPECIALIST_PROVIDER_FOLLOWUP_RESPONSE"
        state["last_followup_action"] = "provider_id_only"
        
        # Stop the graph here so nothing overwrites the state
        requested = interrupt({
            "prompt": state["ai_response"],
            "stage": state["stage"],
        })

        # Store next user message for next turn 
        state["csr_query"] = str(requested).strip()
        return state

    # Specialist flow supports ONLY address inquiry (not assignment)
    if action != "address" or not pid:
        state["ai_response"] = (
            "Please provide a Provider ID and ask for its address (for example: 'address of 12345678')."
        )
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 110
        state["ai_response_type"] = "AURA"
        state["stage"] = "SPECIALIST_NEED_ADDRESS_CLARIFICATION"
        return state

    # Find chosen provider from providers_result
    chosen = None
    for p in providers:
        cand = (
            p.get("providerId")
            or p.get("provId")
            or p.get("ProviderID")
            or p.get("id")
        )
        if str(cand).strip() == str(pid).strip():
            chosen = p
            break

    if not chosen:
        state["ai_response"] = "I couldn't find that Provider ID in the current results. Please pick a Provider ID from the list."
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 110
        state["ai_response_type"] = "AURA"
        state["stage"] = "SPECIALIST_NEED_ADDRESS_CLARIFICATION"
        return state

    prov_id = str(chosen.get("providerId") or pid or "").strip()
    addr_payload = {
        "providerAddress": [{
            "ProviderID": prov_id,
            "AddressType": chosen.get("addressType"),
            "AddressLine1": chosen.get("address1") or chosen.get("addressLine1"),
            "AddressLine2": chosen.get("address2") or chosen.get("addressLine2"),
            "City": chosen.get("city"),
            "State": chosen.get("state"),
            "Zip": chosen.get("zip") or chosen.get("zipCode"),
            "Country": chosen.get("country") or chosen.get("county"),
            "Phone": chosen.get("phone"),
            "Fax": chosen.get("fax"),
            "EffectiveDate": chosen.get("effectiveDate"),
            "TerminationDate": chosen.get("terminationDate"),
        }]
    }

    # Final inquiry response (completed)
    state["ai_response"] = json.dumps(addr_payload, default=str, separators=(",", ":"))
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 110
    state["prompt_title"] = "Do you need further assistance?"
    state["prompts"] = ["Yes", "No"]
    state["stage"] = "COMPLETED"

    requested = interrupt({
        "prompt": state["ai_response"],
        "stage": "SPECIALIST_COMPLETED",
        "prompt_title": state["prompt_title"],
        "prompts": state["prompts"],
        "ai_response_code": state["ai_response_code"],
        "ai_response_type": state["ai_response_type"],
    })

    # store next yes/no answer for routing
    state["csr_query"] = str(requested).strip()
    return state

def node_provider_id_only_warning(state: PCPState) -> PCPState:
    """
    If user sends only Provider ID without asking address/assign,
    show the warning message and STOP here (interrupt).
    Next message resumes back to provider_interaction.
    """
    mark_llm(state)
    msg = "Please check the response that you have provided."

    state["ai_response"] = msg
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "PROVIDER_ID_ONLY_WARNING"

    # IMPORTANT: interrupt each time (no fall-through)
    requested = interrupt({
        "prompt": msg,
        "stage": state["stage"],
    })

    # capture the next user input so provider_interaction can process it
    state["csr_query"] = str(requested).strip()
    state["last_followup_action"] = None
    return state


def node_provider_interaction(state: PCPState) -> PCPState:
    """
    After showing the provider list JSON, this node uses interrupt + LLM
    to interpret whether user wants address or PCP assignment.
    """
    logger.debug("node_provider_interaction")
    providers = state.get("providers_result") or []
    if not providers:
        state["ai_response"] = "No providers are available in the current result. Please start a new search."
        state["prompt_title"] = "No providers found"
        state["prompts"] = []
        state["ai_response_code"] = 404
        state["ai_response_type"] = "Dialog"
        state["stage"] = "ERROR"
        return state

    # Ask user what they want to do next, if csr_query not yet captured
    if not state.get("csr_query"):
        ai_msg = call_horizon(
            "You are a CSR assistant.",
            "The member has seen the provider search JSON. Ask them what they want to do next: "
            "for example, ask for the address of a provider, or assign a specific provider as PCP.",
        )
        mark_llm(state)
        state["ai_response"] = ai_msg
        state["prompt_title"] = "Next action"
        state["prompts"] = []
        state["ai_response_code"] = 108
        state["ai_response_type"] = "Dialog"
        state["stage"] = "PROVIDER_ACTION"

        requested = interrupt({
            "prompt": ai_msg,
            "stage": state["stage"],
        })
        state["csr_query"] = str(requested)

    # Use LLM to understand provider action
    decision = llm_decide_followup_action(state["csr_query"], providers)
    mark_llm(state)
    action = decision.get("action")
    pid = decision.get("provider_id")

    # If user sent ONLY provider id (no text)
    if action == "provider_id_only":
        state["last_followup_action"] = "provider_id_only"
        return state
    
    chosen = None
    if pid:
        for p in providers:
            for k in ("providerId", "provider_id", "id"):
                if str(p.get(k)) == str(pid):
                    chosen = p
                    break
            if chosen:
                break

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

        state["ai_response"] = json.dumps(addr_payload, default=str, separators=(",", ":"))
        state["prompt_title"] = ""
        state["prompts"] = []
        state["ai_response_code"] = 108
        state["ai_response_type"] = "AURA"
        state["stage"] = "PROVIDER_FOLLOWUP_RESPONSE"
        state["last_followup_action"] = "address"
        return state
        
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

    # If other / not clear
    mark_llm(state)
    state["ai_response"] = (
        "I could not understand which provider or action you meant. "
        "Please mention if you want the address of a provider or to assign a provider as PCP, "
        "including the Provider ID."
    )
    state["prompt_title"] = "Clarification"
    state["prompts"] = []
    state["ai_response_code"] = 110
    state["ai_response_type"] = "Dialog"
    state["stage"] = "PROVIDER_FOLLOWUP_RESPONSE"
    state["last_followup_action"] = "other"
    return state

def node_run_provider_search(state: PCPState) -> PCPState:
    logger.debug("node_run_provider_search")

    if state.get("providers_result"):
        return state
    
    providers: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # Case 1 : member knows provider id
    # -----------------------------------------------------
    if state.get("knows_provider"):
        parsed = llm_parse_provider_input(state.get("raw_provider_input") or "")
        search_type = parsed.get("search_type")

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

            # # Use full name for providerName param (if your API uses it)
            # full_provider_name = name or None

            # member_id = state.get("member_id", "")

            # Call normalized name-search service
            mark_api(state, provider_search_by_name)
            providers = provider_search_by_name(
                last_name=last_name or "",
                group_id=state.get("group_id", ""),
                subscriber_id=state.get("subscriber_id", ""),
                member_id="",
                first_name=first_name,
                provider_name="",
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
                is_accpeting = p.get("isAcceptingNewMembers")

                # PCPAssnInd
                pcp_ind = p.get("pcpAssnInd") or p.get("PCPAssnInd")

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
                    "IsAcceptingNewMembers": is_accpeting,
                    "PCPAssnInd": pcp_ind,
                    "DistanceInMiles": dist,
                })

            response_payload = {"providers": grid_list}

            state["providers_result"] = providers
            state["ai_response"] = json.dumps(response_payload, default=str, separators=(",", ":"))
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
            mark_api(state, provider_generic_search)
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

                is_accpeting = p.get("isAcceptingNewMembers")

                pcp_ind = p.get("pcpAssnInd") or p.get("PCPAssnInd")

                dist = p.get("distance_mi") or p.get("distanceInMiles")

                grid_list.append({
                    "ProviderID": str(prov_id),
                    "Name": str(name_val),
                    "Address": addr,
                    "Network": net_str,
                    "IsAcceptingNewMembers": is_accpeting,
                    "PCPAssnInd": pcp_ind,
                    "DistanceInMiles": dist,
                })

            response_payload = {"providers": grid_list}

            state["providers_result"] = providers
            state["ai_response"] = json.dumps(response_payload, default=str, separators=(",", ":"))
            state["prompt_title"] = "Select a provider id to update"
            state["prompts"] = []
            state["ai_response_code"] = 107
            state["ai_response_type"] = "AURA"
            state["stage"] = "SHOW_PROVIDER_LIST"

            requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
            state["csr_query"] = str(requested)
            return state 
        # Provider ID path
        else:
            pid = parsed.get("provider_id") or ""
            state["provider_id"] = pid
            # print("date : ", state.get("asOfDate", ""))
            mark_api(state, provider_search_by_id)
            providers = provider_search_by_id(
                id = pid, 
                group_id = state.get("group_id", ""),
                subscriber_id = state.get("subscriber_id", ""),
                asOfDate = state.get("asOfDate", ""),
                )
            
            # print("raw result : ", providers)
            # Now build the AIResponse JSON in the requried shape using providers
            grid_list: List[Dict[str, Any]] = []
            for p in providers:
                prov_id = (
                    p.get("providerId")
                    or p.get("provId")
                    or p.get("ProviderID")
                    or state.get("provider_id")
                )
                name_val = (
                    p.get("name")
                    or p.get("providerName")
                    or p.get("fullName")
                )

                # Reuse existing helper to build address steing
                addr = _format_address_from_provider(p)
                # print("Address : ", addr)
                # Network in/out
                raw_network = (
                    p.get("networkStatus")
                    or p.get("network")
                    or p.get("Network")
                )
                net_str = ""
                if raw_network:
                    raw_up = str(raw_network).upper()
                    if "IN" in raw_up and "OUT" not in raw_up:
                        net_str = "In Network"
                    elif "OUT" in raw_up:
                        net_str = "Out Network"
                    else:
                        # Fallback to whatever text we have
                        net_str = str(raw_network)
                else:
                    net_str = ""

                is_accpeting = p.get("isAcceptingNewMembers")

                pcp_ind = p.get("pcpAssnInd") or p.get("PCPAssnInd")

                # Distance in miles
                dist = (
                    p.get("distance_mi")
                    or p.get("distanceInMiles")
                    or p.get("Distance_mi")
                )

                grid_list.append({
                    "ProviderID": str(prov_id) if prov_id is not None else "",
                    "Name": str(name_val) if name_val is not None else "",
                    "Address": addr,
                    "Network": net_str,
                    "IsAcceptingNewMembers": is_accpeting,
                    "PCPAssnInd": pcp_ind,
                    "DistanceInMiles": dist
                })

            response_payload = {
                "providers": grid_list
            }

            state["providers_result"] = providers # keep raw list for later steps
            state["ai_response"] = json.dumps(response_payload, default=str, separators=(',', ':'))
            state["prompt_title"] = "Select a provider id to update"
            state["prompts"] = []
            state["ai_response_code"] = 107
            state["ai_response_type"] = "AURA"
            state["stage"] = "SHOW_PROVIDER_LIST"

            # Interrupt to show this JSON grid to UI and wait for next user action
            requested = interrupt({
                "prompt": state["ai_response"],
                "stage": state["stage"],
            })

            # Next user message (e.g., asking for address) will be stored here
            state["csr_query"] = str(requested)
            return state
        
    # -----------------------------------------------------
    # Case 2 : member does not know provider id
    #          (filters path)
    # -----------------------------------------------------
    else:
        # NO flow -> provider_generic_search
        raw_text = (state.get("raw_filter_input") or "").strip()

        group_id = (state.get("group_id") or "").strip()
        subscriber_id = (state.get("subscriber_id") or "").strip()

        if not group_id or not subscriber_id:
            state["ai_response"] = "Member details are missing (groupId/subscriberId). Please start a new conversation."
            state["prompt_title"] = ""
            state["prompts"] = []
            state["ai_response_code"] = 500
            state["ai_response_type"] = "AURA"
            state["stage"] = "ERROR"
            return state

        default_radius = get_default_radius_in_miles(group_id=group_id, practitioner_type="PCP")

        parsed = llm_parse_no_flow_filters(raw_text)

        # Defaults
        radius = default_radius
        provider_lang = ""
        provider_sex = ""
        zip_code = None
        as_of_date = None

        if parsed.get("use_defaults"):
            # keep defaults only
            pass
        else:
            # language
            provider_lang = map_language_to_code(parsed.get("language"))

            # gender
            g = parsed.get("gender")
            if g in ("M", "F"):
                provider_sex = g

            # zip
            z = parsed.get("zip")
            if z and str(z).isdigit() and len(str(z)) == 5:
                zip_code = str(z)

            # date (YYYYMMDD expected from LLM)
            as_of_date = parsed.get("as_of_date")

            # radius override
            r = parsed.get("radius_in_miles")
            if isinstance(r, (int, float)) and r > 0:
                radius = int(r)

        mark_api(state, provider_generic_search)
        providers = provider_generic_search(
            group_id=group_id,
            subscriber_id=subscriber_id,
            radius_in_miles=radius,
            startingLocationZip=zip_code or "",
            asOfDate=as_of_date,                 # service should apply today's date if None/empty
            providerLanguage=provider_lang,      # must be supported in provider_search.py
            providerSex=provider_sex,            # must be supported in provider_search.py
        )        

        # Build SAME grid JSON shape as your other flows
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

        response_payload = {"providers": grid_list}

        state["providers_result"] = providers
        state["ai_response"] = json.dumps(response_payload, default=str, separators=(",", ":"))
        state["prompt_title"] = "Select a provider id to update"
        state["prompts"] = []
        state["ai_response_code"] = 107
        state["ai_response_type"] = "AURA"
        state["stage"] = "SHOW_PROVIDER_LIST"

        requested = interrupt({
            "prompt": state["ai_response"], 
            "stage": state["stage"]
            })
        state["csr_query"] = str(requested)
        return state

def node_run_specialist_generic_search(state: PCPState) -> PCPState:
    """
    Specialist flow:
    - Use stored specialist_service_specialty
    - Parse specialist_raw_filter_input (any combo)
    - Apply defaults (distance from CSV for Specialist + today's YYYYMMDD if date missing)
    - Call provider_generic_search(serviceSpecialty=...)
    - Return provider grid JSON in AIResponse (AURA, 107), PromptTitle/Prompts empty
    """
    logger.debug("node_run_specialist_generic_search")

    # If already have results, don't re-run
    if state.get("providers_result"):
        return state

    group_id = (state.get("group_id") or "").strip()
    subscriber_id = (state.get("subscriber_id") or "").strip()
    specialty = (state.get("specialist_service_specialty") or "").strip()

    if not group_id or not subscriber_id:
        state["ai_response"] = "Member details are missing (groupId/subscriberId). Please start a new conversation."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        return state

    if not specialty:
        state["ai_response"] = "Service specialty code is missing. Please enter the service specialty code."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 109
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ASK_SPECIALTY"
        return state

    # Defaults
    default_radius = get_default_radius_in_miles(group_id=group_id, practitioner_type="Specialist")
    today_yyyymmdd = datetime.now().strftime("%Y%m%d")

    raw_text = (state.get("specialist_raw_filter_input") or "").strip()
    parsed = llm_parse_specialist_filters(raw_text)

    radius = int(default_radius)
    provider_lang = ""
    provider_sex = ""
    zip_code = ""
    as_of_date = today_yyyymmdd

    if parsed.get("use_defaults"):
        # keep defaults only
        pass
    else:
        # language -> providerLanguage
        provider_lang = map_language_to_code(parsed.get("language"))

        # gender -> providerSex (M/F only)
        g = parsed.get("gender")
        if g in ("M", "F"):
            provider_sex = g

        # zip -> startingLocationZip
        z = parsed.get("zip")
        if z and str(z).isdigit() and len(str(z)) == 5:
            zip_code = str(z)

        # date -> asOfDate (YYYYMMDD)
        d = parsed.get("as_of_date")
        if isinstance(d, str) and d.strip():
            as_of_date = d.strip()

        # radius override
        r = parsed.get("radius_in_miles")
        if isinstance(r, (int, float)) and r > 0:
            radius = int(r)

    # IMPORTANT: provider_generic_search must accept these named params
    mark_api(state, provider_generic_search)
    providers = provider_generic_search(
        group_id=group_id,
        subscriber_id=subscriber_id,
        radius_in_miles=radius,
        startingLocationZip=zip_code,
        asOfDate=as_of_date,
        providerLanguage=provider_lang,
        providerSex=provider_sex,
        serviceSpecialty=specialty,
    )    

    # Build grid JSON (same shape)
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

    response_payload = {"providers": grid_list}

    state["providers_result"] = providers
    state["ai_response"] = json.dumps(response_payload, default=str, separators=(",", ":"))
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 107
    state["prompt_title"] = ""
    state["prompts"] = []
    state["stage"] = "SHOW_SPECIALIST_PROVIDER_LIST"

    requested = interrupt({"prompt": state["ai_response"], "stage": state["stage"]})
    state["csr_query"] = str(requested)
    return state

def node_specialist_post_completion(state: PCPState) -> PCPState:
    """
    After specialist address is shown with Yes/No prompts, route:
    - Yes -> route to RETURN_TO_MENU node
    - No  -> closing message + END
    """
    logger.debug("node_specialist_post_completion")

    txt = (state.get("csr_query") or "").strip().lower()

    # Let Horizon interpret yes/no semantically (no hardcoded matching only)
    sys = (
        "You classify whether the user response means YES or NO.\n"
        "Return ONLY JSON: {\"answer\":\"yes\"|\"no\"|\"unknown\"}\n"
    )
    raw = call_horizon(sys, txt).strip()
    mark_llm(state)
    if raw.startswith("```"):
        raw = raw.strip("`")
        i = raw.find("{")
        if i != -1:
            raw = raw[i:]
    ans = "unknown"
    try:
        ans = (json.loads(raw).get("answer") or "unknown").lower()
    except Exception:
        ans = "unknown"

    if ans == "no":
        state["ai_response"] = "We're closing your request-feel free to return if you need anything else."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 101
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "CLOSED"
        return state

    # YES or unknown -> send to menu (same behavior as start)
    # reset specialist-specific fields (keep member details/thread)
    state["providers_result"] = None
    # state["specialist_service_specialty"] = None
    # state["specialist_raw_filter_input"] = None
    state["csr_query"] = ""   # menu will interrupt and refill
    state["stage"] = "GO_MENU"
    return state

def node_return_to_menu(state: PCPState) -> PCPState:
    """
    Emits the START menu payload (AIResponse empty) and interrupts.
    This is used when user says YES after Specialist inquiry completion.
    """
    logger.debug("node_return_to_menu")

    # Pull from config when available (avoid hardcoding text in code)
    menu_title = getattr(settings, "MENU_PROMPT_TITLE", "How can I assist you today?")
    menu_prompts = getattr(settings, "MENU_PROMPTS", None)
    if not isinstance(menu_prompts, list) or not menu_prompts:
        menu_prompts = settings.DEFAULT_PROMPTS

    # Exact UI payload
    state["ai_response"] = ""
    state["ai_response_type"] = "AURA"
    state["ai_response_code"] = 101
    state["prompt_title"] = menu_title
    state["prompts"] = menu_prompts
    state["stage"] = "START"
    mark_llm(state)

    # Interrupt so /chat returns immediately with menu
    requested = interrupt({
        "prompt": "",
        "stage": "RETURN_TO_MENU",
        "prompt_title": menu_title,
        "prompts": menu_prompts,
        "ai_response_code": 101,
        "ai_response_type": "AURA",
    })

    # Resume value becomes the next menu selection
    state["csr_query"] = str(requested).strip()
    return state

def node_wait_next_followup(state: PCPState) -> PCPState:
    """
    Keeps the conversation alive after showing address/warning/etc.
    Interrupts to wait for the next user message, then stores it into csr_query.
    """
    mark_llm(state)
    requested = interrupt({
        "prompt": "",                 # IMPORTANT: empty prompt; UI will display prior ai_response
        "stage": "WAIT_NEXT_FOLLOWUP"
    })

    state["csr_query"] = str(requested)
    return state

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

    print("meme_ck:", meme_ck, "grgr_ck:", grgr_ck, "curr_pid:", curr_pid, "trsn:", trsn)

    if not (meme_ck and grgr_ck and trsn):
        state["ai_response"] = "Missing required member/termination details to update PCP."
        state["ai_response_type"] = "AURA"
        state["ai_response_code"] = 500
        state["prompt_title"] = ""
        state["prompts"] = []
        state["stage"] = "ERROR"
        state["last_followup_action"] = "error"
        return state

    today_dt = datetime.now().date()
    today_mmddyyyy = format_date_mmddyyyy(today_dt)
    tomorrow_dt = today_dt + timedelta(days=1)
    tommorrow_mmddyyyy = format_date_mmddyyyy(tomorrow_dt)
    
    print("today dt : ", today_dt)
    print("today mmddyyyy: ", today_mmddyyyy)
    print("tomorrow dt : ", tomorrow_dt)
    print("tomorrow mmddyyyy: ", tommorrow_mmddyyyy)

    # Termination effective date must come from active PCP
    active_eff = (state.get("active_eff_dt") or "").strip()
    print("active eff: ", active_eff)
    prev_eff_mmddyyyy = format_date_mmddyyyy(active_eff) if active_eff else tommorrow_mmddyyyy
    print("prev eff: ", prev_eff_mmddyyyy)
    print("today dt : ", today_mmddyyyy)
    # Config values (not hardcoded)
    pcp_type = getattr(settings, "pcpType", None) or getattr(settings, "pcp_type", None)
    mctr_orsn = getattr(settings, "mctrOrsn", None) or getattr(settings, "mctr_orsn", None)

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
        if curr_pid != "":
            # print("inside termination if")
            # 1) TERMINATE CURRENT PCP
            term_req = make_change_family_broker_request(
                grgr_ck=grgr_ck,
                meme_ck=meme_ck,
                pcp_type=pcp_type,
                prpr_id=curr_pid,
                eff_dt_mmddyyyy=prev_eff_mmddyyyy,          # from active PCP
                term_dt_mmddyyyy=today_mmddyyyy,            # today
                mctr_trsn=trsn,             # stored termination reason
                mctr_orsn=mctr_orsn,         # from config
            )
            # print("term req : ", term_req)
            term_env = build_executeex_envelope(term_req)
            try:
                mark_api(state, call_execute_ex)
                resp_term = call_execute_ex(term_env)
                print("response terminate : ", resp_term)
            except Exception as te:
                short = extract_short_error(str(te))
                raise RuntimeError(f"Terminate step failed: {short}") from te

        # 2) ADD NEW PCP
        add_req = make_change_family_broker_request(
            grgr_ck=grgr_ck,
            meme_ck=meme_ck,
            pcp_type=pcp_type,
            prpr_id=new_pid,
            eff_dt_mmddyyyy=tommorrow_mmddyyyy,          # tomorrow
            term_dt_mmddyyyy="",                 # empty
            mctr_trsn="",               # empty
            mctr_orsn="",               # empty
        )
        # print("add req : ", add_req)
        add_env = build_executeex_envelope(add_req)
        try:
            mark_api(state, call_execute_ex)
            resp = call_execute_ex(add_env)
            print("resp: ", resp)
        except Exception as ae:
            short = extract_short_error(str(ae))
            raise RuntimeError(f"Assign step failed: {short}") from ae

        # 3) VERIFY
        # mark_api(state, get_active_pcp)
        verify = get_active_pcp(member_key=meme_ck, grgr_ck=grgr_ck)
        active = (verify.get("active") or {}) if isinstance(verify, dict) else {}
        print("active : ", active)
        verified_pid = (
            active.get("provId")
            or active.get("providerId")
            or active.get("PCPProviderId")
        )
        verified_pid = str(verified_pid).strip() if verified_pid else ""
        print("verified pid : ", verified_pid)
        print("new pid : ", new_pid)
        if verified_pid != new_pid:
            raise RuntimeError("PCP update could not be verified. Please try again.")

        # 4) Build confirmation message in PromptTitle (bold values)
        snap = state.get("selected_provider_snapshot") or {}
        name = snap.get("name") or snap.get("providerName") or f"{new_pid}"
        phone = snap.get("phone") or ""
        # full address from your existing formatter
        addr = _format_address_from_provider(snap) if snap else ""

        # Display effective date as MM/DD/YYYY
        eff_disp = format_date_mmddyyyy(tomorrow_dt)
        print("snap : ", snap, " name : ", name, " phone : ", phone, " addr : ", addr," eff disp : ", eff_disp)
        msg = (
            "New PCP has assigned:<br><br>"
            f"Name: <b>{name}</b>.<br>"
            f"Address: <b>{addr}</b>.<br>"
            f"Phome Number: <b>{phone}</b>.<br>"
            f"Effective Date: <b>{eff_disp}</b>.<br><br>"
            "Please allow upto 7 calendar days to receive your new card.<br><br>"
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
    

# ---- Build graph ----

builder = StateGraph(PCPState)

builder.add_node("start", node_start)
builder.add_node("assign_pcp_ask_termination", node_assign_pcp_ask_termination)
builder.add_node("collect_termination_reason", node_collect_termination_reason)
builder.add_node("collect_knows_provider", node_collect_knows_provider)
builder.add_node("collect_provider_input", node_collect_provider_input)
builder.add_node("collect_no_flow_filters", node_collect_no_flow_filters)
builder.add_node("collect_filter_input", node_collect_filter_input)
builder.add_node("run_provider_search", node_run_provider_search)
builder.add_node("provider_interaction", node_provider_interaction)
builder.add_node("specialist_ask_service", node_specialist_ask_service)
builder.add_node("specialist_ask_filters", node_specialist_ask_filters)
builder.add_node("run_specialist_generic_search", node_run_specialist_generic_search)
builder.add_node("specialist_provider_address", node_specialist_provider_address)
builder.add_node("specialist_post_completion", node_specialist_post_completion)
builder.add_node("provider_id_only_warning", node_provider_id_only_warning)
builder.add_node("return_to_menu", node_return_to_menu)
builder.add_node("update_pcp", node_update_pcp)
builder.add_node("wait_next_followup", node_wait_next_followup)
builder.add_node("load_case", node_load_case)

builder.add_edge(START, "load_case")
builder.add_edge("load_case", "start")

# From start, conditionally branch based on menu choice
builder.add_conditional_edges(
    "start",
    path=route_from_menu,
    path_map={
        "assign_pcp": "assign_pcp_ask_termination",
        "specialist": "specialist_ask_service",
        "unsupported": END,
    },
)

builder.add_edge("assign_pcp_ask_termination", "collect_termination_reason")
builder.add_edge("collect_termination_reason", "collect_knows_provider")
builder.add_edge("specialist_ask_service", "specialist_ask_filters")
builder.add_edge("specialist_ask_filters", "run_specialist_generic_search")
builder.add_edge("run_specialist_generic_search", "specialist_provider_address")
# builder.add_edge("specialist_provider_address", "specialist_post_completion")
builder.add_edge("provider_id_only_warning", "provider_interaction")

builder.add_conditional_edges(
    "specialist_provider_address",
    path=route_after_specialist_provider_address,
    path_map={
        "completed": "specialist_post_completion",
        "loop": "specialist_provider_address",
    },
)

builder.add_conditional_edges(
    "collect_knows_provider",
    path=route_knows_provider,
    path_map={
        "knows": "collect_provider_input",
        "unknown": "collect_no_flow_filters",
    },
)

builder.add_edge("collect_no_flow_filters", "run_provider_search")

builder.add_edge("collect_provider_input", "run_provider_search")
builder.add_edge("collect_filter_input", "run_provider_search")

builder.add_edge("run_provider_search", "provider_interaction")

builder.add_conditional_edges(
    "specialist_post_completion",
    path=route_after_specialist_completion,
    path_map={
        "menu": "return_to_menu",
        "end": END,
    },
)

builder.add_conditional_edges(
    "provider_interaction",
    path=route_after_provider_followup,
    path_map={
        "loop": "wait_next_followup",
        "assign": "update_pcp",
        "pid_only": "provider_id_only_warning",
    }
)
builder.add_edge("wait_next_followup", "provider_interaction")
builder.add_edge("update_pcp", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
