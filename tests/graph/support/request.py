from langgraph_openai_serve import GraphRequest


def graph_request(
    model: str,
    *,
    user: str | None = None,
    metadata: dict[str, str] | None = None,
) -> GraphRequest:
    """Build a protocol-neutral request without client tools."""
    return GraphRequest(
        model=model,
        user=user,
        metadata=metadata or {},
        tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
    )
