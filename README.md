# test1

import uuid
import os
import json
import logging
from typing import TypedDict, Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

from datetime import datetime, timezone

# Horizon via OpenRouter (OpenAI-compatible client)
from openai import OpenAI

logger = logging.getLogger("pcp_app")

HORIZON_MODEL = "openrouter/horizon-beta"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set; Horizon calls will fail until configured.")

horizon_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def call_horizon(system_prompt: str, user_prompt: str) -> str:
    resp = horizon_client.chat.completions.create(
        model=HORIZON_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    msg = resp.choices[0].message
    content = msg.content
    if isinstance(content, list):
        # New OpenAI-style response with parts
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- External provider/member APIs (wire to your real services) ----
try:
    from services.provider_search import (
        provider_search_by_id,
        provider_search_by_name,
        provider_generic_search,
    )
    from services.member_rest import get_active_pcp
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
    get_active_pcp = None
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


def node_collect_knows_provider(state: PCPState) -> PCPState:
    logger.debug("➡️ node_collect_knows_provider")
    if state.get("knows_provider") is None:
        ai_msg = call_horizon(
            "You are a CSR assistant.",
            "Ask the member if they already know the provider ID or provider name/city/state they want as PCP. "
            "They should answer Yes or No.",
        )
        state["ai_response"] = ai_msg
        state["prompt_title"] = "Do you know the provider?"
        state["prompts"] = ["Yes", "No"]
        state["ai_response_code"] = 102
        state["ai_response_type"] = "Dialog"
        state["stage"] = "ASK_KNOWS_PROVIDER"

        requested = interrupt({
            "prompt": ai_msg,
            "prompts": state["prompts"],
            "stage": state["stage"],
        })
        answer = str(requested).strip().lower()
        state["csr_query"] = answer
        state["knows_provider"] = "yes" in answer
        return state
    else:
        return state


def node_collect_provider_input(state: PCPState) -> PCPState:
    logger.debug("➡️ node_collect_provider_input")
    if not state.get("raw_provider_input"):
        ai_msg = call_horizon(
            "You are a CSR assistant.",
            "Ask the member to provide either the Provider ID or Provider Name, City and State.",
        )
        state["ai_response"] = ai_msg
        state["prompt_title"] = "Provider details"
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
    logger.debug("➡️ node_run_provider_search")
    providers: List[Dict[str, Any]] = []

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
        else:
            pid = parsed.get("provider_id") or ""
            state["provider_id"] = pid
            providers = run_provider_search_by_id(pid)
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
    addr1 = p.get("address1") or p.get("Address1")
    addr2 = p.get("address2") or p.get("Address2")
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
    phone = p.get("phone") or p.get("Phone")
    if phone:
        lines.append(f"Phone: {phone}")
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
        "assign_pcp": "collect_termination_reason",
        "unsupported": END,
    },
)

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
    title="Conversational PCP Assign (LangGraph + Horizon)",
    version="1.0.0",
    description="Assign PCP flow using LangGraph interrupts and Horizon LLM.",
)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


def config_for_thread(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def base_response(thread_id: str, state: PCPState) -> Dict[str, Any]:
    return {
        "ThreadId": thread_id,
        "AIResponse": state.get("ai_response", ""),
        "AIResponseDateTime": now_str(),
        "CurrentStage": state.get("stage"),
        "PromptTitle": state.get("prompt_title") or "",
        "Prompts": state.get("prompts") or [],
        "AIResponseCode": state.get("ai_response_code"),
        "AIResponseType": state.get("ai_response_type"),
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
            return JSONResponse(base_response(thread_id, last_state))
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
            return JSONResponse(base_response(req.thread_id, last_state))
        for _, value in event.items():
            last_state = value

    # Completed
    return JSONResponse(base_response(req.thread_id, last_state))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

