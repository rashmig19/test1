from langgraph.types import interrupt

# make sure get_active_pcp is imported:
# from services.member_rest import get_active_pcp

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
    logger.debug("➡️ node_assign_pcp_ask_termination, state=%s", state)

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
            active_pcp = get_active_pcp(member_id)
        except Exception as ex:
            logger.exception("get_active_pcp failed: %s", ex)
            state["ai_response"] = "Unable to fetch your current PCP details right now. Please try again later."
            state["ai_response_code"] = 500
            state["ai_response_type"] = "Dialog"
            state["prompts"] = []
            state["stage"] = "ERROR"
            return state

        # 2) Extract Provider ID from response (adjust key as per your API)
        #    example keys: 'providerId', 'pcpProviderId', etc.
        active_provider_id = (
            active_pcp.get("providerId")
            or active_pcp.get("pcpProviderId")
            or active_pcp.get("PCPProviderId")
        )
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
        state["ai_response_code"] = 112          # <-- as you requested
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
