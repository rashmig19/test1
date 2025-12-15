# After user says YES in specialist completion, we show menu via return_to_menu.
# IMPORTANT: once return_to_menu captures the next menu choice (resume value),
# we must route from it, otherwise graph will keep resuming on return_to_menu forever.
builder.add_conditional_edges(
    "return_to_menu",
    path=route_from_menu,
    path_map={
        "assign_pcp": "assign_pcp_ask_termination",
        "specialist": "specialist_ask_service",
        "unsupported": END,
    },
)
