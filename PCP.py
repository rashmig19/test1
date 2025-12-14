interaction_id: str

    # raw + filtered case payload for later use
    case_raw: Optional[Dict[str, Any]]
    case_filtered: Optional[Dict[str, Any]]

#################################################

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
        mark_api(state, get_case_by_interaction_id)
        raw = get_case_by_interaction_id(interaction_id)
        filtered = strip_px_keys(raw, prefix="px")

        state["case_raw"] = raw
        state["case_filtered"] = filtered
        return state

    except Exception as ex:
        logger.exception("Get Case API failed: %s", ex)
        # Do NOT break the existing PCP flow; just store empty and continue
        state["case_raw"] = None
        state["case_filtered"] = None
        return state


#######################################################################

builder.add_node("load_case", node_load_case)

builder.add_edge(START, "load_case")
builder.add_edge("load_case", "start")


##################################################

@app.get("/init-conversation")
def init_conversation(member_id: str, InteractionID: str):

initial_state: PCPState = {
    "thread_id": thread_id,
    "member_id": member_id.strip(),
    "interaction_id": (InteractionID or "").strip(),
    "stage": "",
    "csr_query": "",
    "ai_response": "",
    "prompts": [],
}

