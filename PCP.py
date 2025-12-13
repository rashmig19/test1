def route_after_specialist_provider_address(state: PCPState) -> str:
    # Only go to post-completion logic if we truly completed address inquiry
    if state.get("stage") == "COMPLETED":
        return "completed"
    return "loop"
=================================
builder.add_conditional_edges(
    "specialist_provider_address",
    path=route_after_specialist_provider_address,
    path_map={
        "completed": "specialist_post_completion",
        "loop": "specialist_provider_address",
    },
)
============================
