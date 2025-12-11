import uuid
import os
import json
import logging
from typing import TypedDict, Optional, List, Dict, Any, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

from datetime import datetime, timezone

import requests
from config import settings
from requests.adapters import HTTPAdapter, Retry
import time

from services.member_rest import get_active_pcp, member_search

logger = logging.getLogger("pcp_app")

# ------------------------------------------------------------------------------
# Config knobs (read from env, with sane defaults)
# ------------------------------------------------------------------------------
# Base URL to Horizon Gateway (no trailing slash)
HORIZON_GATEWAY = os.getenv("HORIZON_GATEWAY", settings.HORIZON_GATEWAY).rstrip("/")

# OAuth client creds
HORIZON_CLIENT_ID = os.getenv("HORIZON_CLIENT_ID", settings.HORIZON_CLIENT_ID)
HORIZON_CLIENT_SECRET = os.getenv("HORIZON_CLIENT_SECRET", settings.HORIZON_CLIENT_SECRET)
verify_val = settings.CA_BUNDLE_PATH if settings.VERIFY_SSL_SOAP else False
# verify = settings.VERIFY_SSL_REST
# Timeouts & retry policy
DEFAULT_TIMEOUT = float(os.getenv("HORIZON_TIMEOUT_SECONDS", "30"))
RETRY_TOTAL = int(os.getenv("HORIZON_RETRY_TOTAL", "3"))
RETRY_BACKOFF = float(os.getenv("HORIZON_RETRY_BACKOFF", "0.5"))

# Horizon via OpenRouter (OpenAI-compatible client)
#from openai import OpenAI

# ------------------------------------------------------------------------------
# Token cache (memory)
# ------------------------------------------------------------------------------
_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,  # epoch seconds
}

# Simple in-memory token cache so we don’t call auth on every request
_member_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,   # epoch seconds
}

def _session_with_retries() -> requests.Session:
    """Requests session with basic retry policy suitable for gateways."""
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session



def _auth_endpoint() -> str:
    """Build Horizon OAuth token endpoint."""
    if not HORIZON_GATEWAY:
        raise ValueError("HORIZON_GATEWAY is not set")
    # Adjust if your Horizon auth path differs:
    return f"{HORIZON_GATEWAY}/oauth2/token"

def getAuthToken(client_id: str, client_secret: str, address: str) -> str:
    """
    Get (and cache) a Horizon Bearer token via client_credentials.
    - Respects in-memory cache until expiry (with a small safety margin).
    - Set env HORIZON_VERIFY_SSL to 'false' for dev self-signed, or to a CA bundle path.
    """
    # If a valid, non-expired token is cached, return it
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 10:
        return _token_cache["access_token"]

    # Validate inputs
    if not address:
        raise ValueError("Horizon address is required (HORIZON_GATEWAY).")
    if not client_id or not client_secret:
        raise ValueError("Horizon client_id/client_secret are required.")

    url = _auth_endpoint()
    # Most gateways accept standard OAuth2 client_credentials form body
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    
    session = _session_with_retries()
    try:
        resp = session.post(url, data=data, headers=headers, timeout=DEFAULT_TIMEOUT, verify=verify_val)
        resp.raise_for_status()
        payload = resp.json()

        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)  # seconds
        if not access_token:
            raise ValueError(f"Token endpoint missing 'access_token'. Response: {payload}")

        _token_cache["access_token"] = access_token
        _token_cache["expires_at"] = time.time() + float(expires_in)
        return access_token

    except requests.RequestException as e:
        logger.error("Horizon token request failed: %s", e, exc_info=True)
        raise
    except ValueError as e:
        logger.error("Invalid token response: %s", e, exc_info=True)
        raise

def call_horizon(system_prompt: str, user_prompt: str) -> str:
    auth_token = getAuthToken(settings.HORIZON_CLIENT_ID, settings.HORIZON_CLIENT_SECRET, settings.HORIZON_GATEWAY)
    url = f"{settings.HORIZON_CHAT_ENDPOINT}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60, verify=verify_val)
    resp.raise_for_status()
    data=resp.json()
    # print(data)

    content = None
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")

            if content is None and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                m = choice.get("message") or {}
                content = m.get("content")
            if content is None and isinstance(data.get("text"), str):
                content = data["text"]

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    if isinstance(content, str):
        return content.strip()
    
    return str(data)
    
def now_str() -> str:
    return datetime.now().strftime("%m/%d/%Y %I:%M %p")


# ---- External provider/member APIs (wire to your real services) ----
try:
    from services.provider_search import (
        provider_search_by_id,
        provider_search_by_name,
        provider_generic_search,
    )
    
    from services.soap_client import (
        make_change_family_broker_request,
        build_executeex_envelope,
        call_execute_ex,
        extract_short_error,
    )
except Exception:
    provider_search_by_id = None
    provider_search_by_name = None
    provider_generic_search = None
    # get_active_pcp = None
    make_change_family_broker_request = None
    build_executeex_envelope = None
    call_execute_ex = None
    extract_short_error = None


# ---- Graph state ----

class PCPState(TypedDict, total=False):
    # Core conversation info
    thread_id: str
    member_id: str
    stage: str
    csr_query: str
    last_ai_message: str

    # Data captured along the flow
    termination_reason: Optional[str]
    knows_provider: Optional[bool]

    # Raw user inputs
    raw_provider_input: Optional[str]
    raw_filter_input: Optional[str]

    # Parsed provider search params
    provider_id: Optional[str]
    provider_name: Optional[str]
    provider_city: Optional[str]
    provider_state: Optional[str]
    language: Optional[str]
    radius_in_miles: Optional[int]
    gender: Optional[str]

    # Provider results and selection
    providers_result: Optional[List[Dict[str, Any]]]
    last_selected_provider_id: Optional[str]

    # Member details
    active_provider_id: Optional[str]
    meme_ck: Optional[str]
    grgr_ck: Optional[str]
    group_id: Optional[str]
    subscriber_id: Optional[str]

    # Response formatting for UI
    ai_response: str
    prompts: List[str]
    ai_response_code: Optional[int]
    ai_response_type: Optional[str]
    prompt_title: Optional[str]


DEFAULT_PROMPTS = [
    "Assign PCP",
    "Search for specialist",
    "Need something else",
]


# ---- Helper wrappers around your tools ----

def run_provider_search_by_id(pid: str) -> List[Dict[str, Any]]:
    if provider_search_by_id is None:
        # dummy fallback so file runs even without backend
        return [{"providerId": pid, "name": f"Dummy Provider {pid}"}]
    return provider_search_by_id(provider_id=pid)


def run_provider_search_by_name(name: str, city: str, state: str) -> List[Dict[str, Any]]:
    if provider_search_by_name is None:
        return [{
            "providerId": "99999",
            "name": name or "Dummy Name",
            "city": city,
            "state": state,
        }]
    return provider_search_by_name(name=name, city=city, state=state)


def run_provider_generic_search(member_id: str, language: Optional[str],
                                radius: Optional[int], gender: Optional[str]) -> List[Dict[str, Any]]:
    if provider_generic_search is None:
        return [{
            "providerId": "88888",
            "name": "Generic PCP",
            "city": "Sample City",
            "state": "ST",
        }]
    return provider_generic_search(
        member_id=member_id,
        language=language,
        radius_in_miles=radius,
        gender=gender,
    )


def perform_pcp_update(member_id: str, provider_id: str, termination_reason: str) -> str:
    """
    Wrap your existing SOAP/member logic here.
    For now we simulate success if real services are missing.
    """
    if get_active_pcp is None or make_change_family_broker_request is None:
        return f"PCP updated successfully to provider {provider_id} for member {member_id}."

    try:
        active_pcp = get_active_pcp(member_id)

        # terminate old
        term_req = make_change_family_broker_request(
            member_id=member_id,
            provider_id=active_pcp.get("providerId"),
            termination_reason=termination_reason,
            action="TERMINATE",
        )
        term_env = build_executeex_envelope(term_req)
        term_resp = call_execute_ex(term_env)
        err = extract_short_error(term_resp)
        if err:
            raise RuntimeError(f"Termination failed: {err}")

        # add new
        add_req = make_change_family_broker_request(
            member_id=member_id,
            provider_id=provider_id,
            termination_reason=None,
            action="ADD",
        )
        add_env = build_executeex_envelope(add_req)
        add_resp = call_execute_ex(add_env)
        err = extract_short_error(add_resp)
        if err:
            raise RuntimeError(f"Add PCP failed: {err}")

        return f"PCP updated successfully to provider {provider_id}."
    except Exception as ex:
        logger.exception("Error during PCP update: %s", ex)
        raise


# ---- LLM-based parsers (no regex helpers) ----

def llm_parse_provider_input(user_text: str) -> Dict[str, Any]:
    """
    Ask Horizon to interpret the user's provider details.
    The model MUST respond with JSON only.
    """
    system = (
        "You are an AI that extracts provider search parameters from free text. "
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "search_type": "id" | "name_city_state",\n'
        '  "provider_id": string | null,\n'
        '  "name": string | null,\n'
        '  "city": string | null,\n'
        '  "state": string | null\n'
        "}\n"
        "If you are given a numeric provider id, use search_type='id'. "
        "If you are given name, city, state, use search_type='name_city_state'."
    )
    raw = call_horizon(system, user_text)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        data = json.loads(raw)
        return data
    except Exception:
        logger.warning("Failed to parse provider input JSON from LLM: %s", raw)
        return {
            "search_type": "id",
            "provider_id": user_text.strip(),
            "name": None,
            "city": None,
            "state": None,
        }


def llm_parse_filter_input(user_text: str) -> Dict[str, Any]:
    system = (
        "You are an AI that extracts PCP search filters from free text. "
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "language": string | null,\n'
        '  "radius_in_miles": number | null,\n'
        '  "gender": "M" | "F" | null\n'
        "}\n"
        "Pick gender from M/F when user mentions male/female, man/woman, etc."
    )
    raw = call_horizon(system, user_text)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        data = json.loads(raw)
        return data
    except Exception:
        logger.warning("Failed to parse filter JSON from LLM: %s", raw)
        return {"language": None, "radius_in_miles": None, "gender": None}


def llm_decide_provider_action(user_text: str, providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask Horizon whether user wants address or assign PCP, and for which provider.
    """
    system = (
        "You are an assistant that interprets a member's request about a provider search result.\n"
        "You receive the user's message and a list of providers with providerId.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "action": "address" | "assign_pcp" | "other",\n'
        '  "provider_id": string | null\n'
        "}\n"
        "If the user asks for address/location of a provider, action='address'.\n"
        "If the user asks to assign a provider as PCP, action='assign_pcp'."
    )
    providers_brief = []
    for p in providers:
        providers_brief.append({
            "providerId": p.get("providerId") or p.get("provider_id") or p.get("id"),
            "name": p.get("name"),
            "city": p.get("city"),
            "state": p.get("state"),
        })
    user_payload = {
        "message": user_text,
        "providers": providers_brief,
    }
    raw = call_horizon(system, json.dumps(user_payload))
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    try:
        data = json.loads(raw)
        return data
    except Exception:
        logger.warning("Failed to parse provider action JSON from LLM: %s", raw)
        return {"action": "other", "provider_id": None}


# ---- Graph nodes ----

def node_start(state: PCPState) -> PCPState:
    """
    Initial node: show main menu and interrupt to get user's choice.
    """
    logger.debug("➡️ node_start")
    menu_text = (
        "You can choose one of the following options:\n"
        "1. Assign PCP\n"
        "2. Search for specialist\n"
        "3. Need something else"
    )
    ai_msg = call_horizon(
        "You are a helpful CSR assistant. Ask the member to choose one of the options.",
        menu_text,
    )
    state["ai_response"] = ai_msg
    state["prompt_title"] = "How can I assist you today?"
    state["prompts"] = DEFAULT_PROMPTS
    state["ai_response_code"] = 100
    state["ai_response_type"] = "Dialog"
    state["stage"] = "MENU"

    # Interrupt, waiting for user choice
    requested = interrupt({
        "prompt": ai_msg,
        "prompts": DEFAULT_PROMPTS,
        "stage": state["stage"],
    })

    # When resumed:
    state["csr_query"] = str(requested)
    return state


def node_collect_termination_reason(state: PCPState) -> PCPState:
    logger.debug("➡️ node_collect_termination_reason")
    if not state.get("termination_reason"):
        ai_msg = call_horizon(
            "You are a CSR assistant.",
            "Ask the member to provide a termination reason for their current PCP.",
        )
        state["ai_response"] = ai_msg
        state["prompt_title"] = "Termination reason"
        state["prompts"] = []
        state["ai_response_code"] = 101
        state["ai_response_type"] = "Dialog"
        state["stage"] = "ASK_TERMINATION"

        requested = interrupt({
            "prompt": ai_msg,
            "stage": state["stage"],
        })
        state["termination_reason"] = str(requested)
        return state
    else:
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
            print("active_pcp: ", active_pcp)
            curr_active = active_pcp.get("active") or {}
        except Exception as ex:
            logger.exception("get_active_pcp failed: %s", ex)
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
        print("active_provider_id: ", active_provider_id)
        state["active_provider_id"] = str(active_provider_id) if active_provider_id else ""

        # 3) Build question including active provider id
        base_question = f"Please select the termination reason for current PCP - {state['active_provider_id']}."

        # 4) Optional: pass through Horizon LLM to keep your 'LLM brain' requirement
        ai_msg = call_horizon(
            "You are a CSR assistant. Ask the member to select a termination reason using the given sentence. "
            "Keep the provider ID exactly as it is.",
            base_question,
        )

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
        state["termination_reason"] = str(requested)
        logger.debug("Termination reason captured from interrupt: %s", state["termination_reason"])
        return state

    # ----------------------------------------------------------------------
    # Else, termination_reason is already filled (we resumed this node),
    # so we just handoff to the next stage in your flow (ask provider id / yes-no etc.)
    # ----------------------------------------------------------------------
    return state

def node_collect_knows_provider(state: PCPState) -> PCPState:
    logger.debug("node_collect_knows_provider")
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
        # ai_msg = call_horizon(
        #     "You are a CSR assistant.",
        #     "Ask the member if they already know the provider ID or provider name/city/state they want as PCP. "
        #     "They should answer Yes or No.",
        # )
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
            "The message MUST:\n"
            "- Start with a bold line: 'Ask the below questions:'\n"
            "- Then show three bullet points (or dotted list items) in this exact order and wording:\n"
            " 1) 'May I know the Provider ID? or Provider Name, City and State?'\n"
            " 2) 'Search would be performed based on member's home address. If you would like to search provider at different location, please provide with zip code and address line 1.'\n"
            " 3) 'Search will be performed with today's date unless a different date is provided (MM-DD-YYYY).'\n"
            "Use simple markdown-style formatting: bold for the first line, and each question on its own line starting with a bullet or a dot. "
            "Return ONLY the formatted text, no explanations."
        )
        user_prompt = "Generate the dialog text exactly as specified, to be displayed to the CSR as instructions."
        # ai_msg = call_horizon(
        #     "You are a CSR assistant.",
        #     "Ask the member to provide either the Provider ID or Provider Name, City and State.",
        # )
        ai_msg = call_horizon(system_prompt, user_prompt).strip()

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


def node_collect_filter_input(state: PCPState) -> PCPState:
    logger.debug("➡️ node_collect_filter_input")
    if not state.get("raw_filter_input"):
        ai_msg = call_horizon(
            "You are a CSR assistant.",
            "Ask the member for any of these PCP search filters: language preference, radius in miles, gender. "
            "They may provide all or any subset.",
        )
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
            name = parsed.get("name") or ""
            city = parsed.get("city") or ""
            st = parsed.get("state") or ""
            state["provider_name"] = name
            state["provider_city"] = city
            state["provider_state"] = st
            providers = run_provider_search_by_name(name, city, st)
        # Provider ID path
        else:
            pid = parsed.get("provider_id") or ""
            state["provider_id"] = pid
            print("date : ", state.get("asOfDate", ""))
            raw_result = run_provider_search_by_id(
                id = pid, 
                group_id = state.get("group_id", ""),
                subscriber_id = state.get("subscriber_id", ""),
                asOfDate = state.get("asOfDate", ""),
                )

            # Normalize provider list into requried JSON structure
            normalized_list : List[Dict[str, Any]] = []

            if isinstance(raw_result, dict) and raw_result.get("providerDetails"):
                details = raw_result.get("providerDetails") or []
                if isinstance(details, dict):
                    details = [details]

                for d in details:
                    info = d.get("providerInfo") or {}
                    contact = d.get("providerContact") or {}

                    prov_id = info.get("providerId") or pid
                    name_val =(
                        info.get("providerName")
                        or info.get("providerFullName")
                    )

                    flat_p = {
                        "providerId": prov_id,
                        "name": name_val,

                        # Address fields for address formatting & later use
                        "address1": contact.get("addressLine1"),
                        "address2": contact.get("addressLine2"),
                        "city": contact.get("city"),
                        "state": contact.get("state"),
                        "zip": contact.get("zip"),
                        "county": contact.get("county"),
                        "phone": contact.get("phone"),
                        "fax": contact.get("fax"),

                        # Network / flags / distance
                        "networkStatus": info.get("networkStatus"),
                        "acceptingNewMembers": info.get("acceptingNewMembers"),
                        "pcpAssnInd": info.get("pcpAssnInd"),
                        "distance_mi": info.get("distanceInMiles"),
                    }

                    normalized_list.append(flat_p)

                providers = normalized_list
            
            else:
                # Fallback if API returned a flat list already
                if isinstance(raw_result, list):
                    providers = raw_result
                else:
                    providers = [raw_result] if raw_result else []

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

                # Is accpeting new members
                accept_val = (
                    p.get("isAcceptingNewMembers")
                    or p.get("acceptingNewPatients")
                    or p.get("AcceptingNewMembers")
                )
                is_accpeting = "Y" if str(accept_val).upper() in ("Y", "YES", "TRUE", "1") else "N"
                
                # PCP assignment indicator
                pcp_val = p.get("pcpAssnInd") or p.get("PCPAssnInd")
                pcp_ind = "Y" if str(pcp_val).upper() in ("Y", "YES", "TRUE", "1") else "N"

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
                    "AcceptingNewMembers": is_accpeting,
                    "PCPAssignmentIndicator": pcp_ind,
                    "DistanceInMiles": dist
                })

            response_payload = {
                "providers": grid_list
            }

            state["providers_result"] = providers # keep raw list for later steps
            state["ai_response"] = json.dumps(response_payload, indent=2, default=str)
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
        parsed = llm_parse_filter_input(state.get("raw_filter_input") or "")
        state["language"] = parsed.get("language")
        radius_val = parsed.get("radius_in_miles")
        if isinstance(radius_val, (int, float)):
            state["radius_in_miles"] = int(radius_val)
        state["gender"] = parsed.get("gender")

        providers = run_provider_generic_search(
            member_id=state["member_id"],
            language=state["language"],
            radius=state.get("radius_in_miles"),
            gender=state["gender"],
        )

    state["providers_result"] = providers
    # Requirement: show raw JSON in AIResponse
    state["ai_response"] = json.dumps(providers, indent=2, default=str)
    state["prompt_title"] = "Provider search results"
    state["prompts"] = []
    state["ai_response_code"] = 107
    state["ai_response_type"] = "JSON"
    state["stage"] = "SHOW_PROVIDER_LIST"
    return state


def _format_address_from_provider(p: Dict[str, Any]) -> str:
    lines: List[str] = []
    addr1 = p.get("addressLine1") or p.get("Address1")
    addr2 = p.get("addressLine2") or p.get("Address2")
    if addr1:
        lines.append(str(addr1))
    if addr2:
        lines.append(str(addr2))
    city = p.get("city") or p.get("City")
    state_val = p.get("state") or p.get("State")
    zipcode = p.get("zip") or p.get("Zip") or p.get("zipCode")
    city_line_parts = [x for x in [city, state_val, zipcode] if x]
    if city_line_parts:
        lines.append(", ".join(city_line_parts))

    county = p.get("county") or p.get("County")
    if county:
        lines.append(f"Country: {county}")

    phone = p.get("phone") or p.get("Phone")
    if phone:
        lines.append(f"Phone: {phone}")
    
    fax= p.get("fax") or p.get("Fax")
    if fax:
        lines.append(f"Fax: {fax}")

    return "\n".join(lines) if lines else "Address details not available."


def node_provider_interaction(state: PCPState) -> PCPState:
    """
    After showing the provider list JSON, this node uses interrupt + LLM
    to interpret whether user wants address or PCP assignment.
    """
    logger.debug("➡️ node_provider_interaction")
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
    decision = llm_decide_provider_action(state["csr_query"], providers)
    action = decision.get("action")
    pid = decision.get("provider_id")

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
        addr = _format_address_from_provider(chosen)
        state["ai_response"] = addr
        state["prompt_title"] = "Provider address"
        state["prompts"] = []
        state["ai_response_code"] = 109
        state["ai_response_type"] = "Dialog"
        state["stage"] = "SHOW_PROVIDER_ADDRESS"
        # clear csr_query so next /chat can ask another action on same list
        state["csr_query"] = ""
        return state

    if action == "assign_pcp" and chosen:
        pid_final = chosen.get("providerId") or chosen.get("provider_id") or pid
        state["last_selected_provider_id"] = str(pid_final)

        try:
            confirmation = perform_pcp_update(
                member_id=state["member_id"],
                provider_id=str(pid_final),
                termination_reason=state.get("termination_reason") or "",
            )
        except Exception as ex:
            state["ai_response"] = f"An error occurred while updating PCP: {ex}"
            state["prompt_title"] = "Error"
            state["prompts"] = []
            state["ai_response_code"] = 500
            state["ai_response_type"] = "Dialog"
            state["stage"] = "ERROR"
            return state

        provider_name = chosen.get("name") or chosen.get("Name") or f"Provider {pid_final}"
        addr = _format_address_from_provider(chosen)
        base_msg = (
            "New PCP has been assigned.\n"
            f"Name: {provider_name}\n"
            f"Address: {addr}\n"
            f"Provider ID: {pid_final}\n\n"
            "Please allow up to 7 calendar days to receive your new card.\n\n"
            "Do you need further assistance?"
        )

        ai_msg = call_horizon(
            "You are a CSR assistant. Rephrase the PCP confirmation message clearly, but keep the same information.",
            base_msg,
        )
        state["ai_response"] = ai_msg
        state["prompt_title"] = "PCP update confirmation"
        state["prompts"] = ["Yes", "No"]
        state["ai_response_code"] = 101
        state["ai_response_type"] = "Dialog"
        state["stage"] = "COMPLETED"
        return state

    # If other / not clear
    state["ai_response"] = (
        "I could not understand which provider or action you meant. "
        "Please mention if you want the address of a provider or to assign a provider as PCP, "
        "including the Provider ID."
    )
    state["prompt_title"] = "Clarification"
    state["prompts"] = []
    state["ai_response_code"] = 110
    state["ai_response_type"] = "Dialog"
    state["stage"] = "PROVIDER_ACTION"
    state["csr_query"] = ""  # so next call will re-interrupt
    return state


# ---- Conditional routing helpers (for conditional edges) ----

def route_from_menu(state: PCPState) -> str:
    text = (state.get("csr_query") or "").lower()
    if "assign" in text and "pcp" in text:
        return "assign_pcp"
    # Search for specialist etc. can be added later
    return "unsupported"


def route_knows_provider(state: PCPState) -> str:
    if state.get("knows_provider"):
        return "knows"
    return "unknown"


# ---- Build graph ----

builder = StateGraph(PCPState)

builder.add_node("start", node_start)
builder.add_node("assign_pcp_ask_termination", node_assign_pcp_ask_termination)
builder.add_node("collect_termination_reason", node_collect_termination_reason)
builder.add_node("collect_knows_provider", node_collect_knows_provider)
builder.add_node("collect_provider_input", node_collect_provider_input)
builder.add_node("collect_filter_input", node_collect_filter_input)
builder.add_node("run_provider_search", node_run_provider_search)
builder.add_node("provider_interaction", node_provider_interaction)

builder.add_edge(START, "start")

# From start, conditionally branch based on menu choice
builder.add_conditional_edges(
    "start",
    path=route_from_menu,
    path_map={
        "assign_pcp": "assign_pcp_ask_termination",
        "unsupported": END,
    },
)

builder.add_edge("assign_pcp_ask_termination", "collect_knows_provider")
builder.add_edge("collect_termination_reason", "collect_knows_provider")

builder.add_conditional_edges(
    "collect_knows_provider",
    path=route_knows_provider,
    path_map={
        "knows": "collect_provider_input",
        "unknown": "collect_filter_input",
    },
)

builder.add_edge("collect_provider_input", "run_provider_search")
builder.add_edge("collect_filter_input", "run_provider_search")
builder.add_edge("run_provider_search", "provider_interaction")
builder.add_edge("provider_interaction", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ---- FastAPI wiring with interrupts ----

app = FastAPI(
    title="Provider Search",
    version="1.0.0",
    description="API for assigning PCP",
)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


def config_for_thread(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def base_response(
        thread_id: str,
        ai_response: str,
        stage: str,
        csr_query: str = "",
        prompts: Optional[List[str]] = None,
        ai_response_code: int = 101,
        ai_response_type: str = "AURA",
        prompt_title: str ="How can I assist you today?",
        api_status: Literal["success", "error"] = "success",
) -> dict:
    return {
        "APIStatus": api_status,
        "APISessionId": thread_id,
        "CSRQuery": csr_query,
        "AIResponse": ai_response,
        "AIResponseType": ai_response_type,
        "AIResponseCode": ai_response_code,
        "AIResponseDateTime": now_str(),
        "CurrentStage": stage,
        "PromptTitle": prompt_title,
        "Prompts": prompts,
    }


@app.get("/init-conversation")
def init_conversation(member_id: str):
    """
    Start the assign PCP conversation:
    - member_id is taken as a query parameter (GET)
    - Creates a new thread_id
    - Runs graph until the first interrupt
    - Returns AIResponse + Prompts + CurrentStage
    """
    if not member_id:
        raise HTTPException(status_code=400, detail="member_id is required")

    thread_id = str(uuid.uuid4())
    cfg = config_for_thread(thread_id)

    initial_state: PCPState = {
        "thread_id": thread_id,
        "member_id": member_id.strip(),
        "stage": "",
        "csr_query": "",
        "ai_response": "",
        "prompts": [],
    }

    last_state: PCPState = initial_state
    for event in graph.stream(initial_state, cfg):
        if "__interrupt__" in event:
            # Node already populated ai_response etc.
            for k, value in event.items():
                if k != "__interrupt__":
                    last_state = value
            return JSONResponse(
                base_response(
                    thread_id=thread_id,
                    ai_response= "",
                    stage = "START",
                    csr_query= "",
                    prompts=DEFAULT_PROMPTS,
                    ai_response_code = 101,
                    ai_response_type="AURA",
                    prompt_title="How can I assist you today?",
                    )
                )
        for _, value in event.items():
            last_state = value

    return JSONResponse(base_response(thread_id, last_state))


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Resume the conversation using LangGraph interrupt resume:
    - Uses Command(resume=message)
    - Runs until next interrupt or completion
    """
    if not req.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    cfg = config_for_thread(req.thread_id)
    last_state: PCPState = {}

    resume_cmd = Command(resume=req.message)

    for event in graph.stream(resume_cmd, cfg):
        if "__interrupt__" in event:
            for k, value in event.items():
                if k != "__interrupt__":
                    last_state = value
            
            interrupt_payloads = event["__interrupt__"]
            interrupt_payload = interrupt_payloads[0].value if interrupt_payloads else {}
            prompt = interrupt_payload.get("prompt", "")
            stage_from_node = interrupt_payload.get("stage", last_state.get("stage", ""))

            if stage_from_node == "WAIT_TERMINATION_REASON":
                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage = "WAIT_TERMINATION_REASON",
                        ai_response=prompt,
                        csr_query=req.message or "",
                        prompts=[],
                        prompt_title="",
                        ai_response_code=112,
                        ai_response_type="AURA",
                        )
                    )
            
            elif stage_from_node == "ASK_KNOWS_PROVIDER":
                question_title = prompt or last_state.get("prompt_title") or ""
                prompts_list = last_state.get("prompts") or ["Yes", "No"]

                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage = "ASK_KNOWS_PROVIDER",
                        ai_response="",
                        csr_query=req.message or "",
                        prompts=prompts_list,
                        prompt_title=question_title,
                        ai_response_code=101,
                        ai_response_type="AURA",
                        )
                    )
            
            elif stage_from_node == "ASK_PROVIDER_INPUT":
                ai_resp_text = prompt or last_state.get("ai_response", "")

                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage = "ASK_PROVIDER_INPUT",
                        ai_response=ai_resp_text,
                        csr_query=req.message or "",
                        prompts=[],
                        prompt_title="",
                        ai_response_code=103,
                        ai_response_type="Dialog",
                        )
                    )
            
            # Provider List (ID search) - show JSON grid to user
            elif stage_from_node == "SHOW_PROVIDER_LIST":
                ai_resp_text = prompt or last_state.get("ai_response", "")

                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage = "SHOW_PROVIDER_LIST",
                        ai_response=ai_resp_text,
                        csr_query=req.message or "",
                        prompts=[],
                        prompt_title="Select a provider id to update",
                        ai_response_code=107,
                        ai_response_type="AURA",
                        )
                    )
            
            # Default handling for any other interrupts
            else:
                return JSONResponse(
                    base_response(
                        thread_id=req.thread_id,
                        stage = stage_from_node or last_state.get("stage", "") or "",
                        ai_response=last_state.get("ai_response", ""),
                        csr_query=req.message or "",
                        prompts=last_state.get("prompts", []),
                        prompt_title=last_state.get("prompt_title", ""),
                        ai_response_code=last_state.get("ai_response_code", 101),
                        ai_response_type=last_state.get("ai_response_type", "AURA"),
                        )
                    )
        
        for _, value in event.items():
            last_state = value

    # Completed
    return JSONResponse(
        base_response(
            thread_id=req.thread_id,
            stage = last_state.get("stage", ""),
            ai_response=last_state.get("ai_response", ""),
            csr_query=req.message or "",
            prompts=last_state.get("prompts", []),
            prompt_title=last_state.get("prompt_title", ""),
            ai_response_code=last_state.get("ai_response_code", 101),
            ai_response_type=last_state.get("ai_response_type", "AURA"),
            )
        )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
