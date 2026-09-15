"""Graph-registration helpers shared by API behavior tests."""

from langgraph_openai_serve import GraphConfig, GraphRegistry


def replace_graph_config(
    registry: GraphRegistry,
    model_id: str,
    **changes: object,
) -> GraphConfig:
    """Replace one immutable config after validating the complete new value."""
    current = registry.get_graph(model_id)
    values = {
        field_name: getattr(current, field_name)
        for field_name in GraphConfig.model_fields
    }
    replacement = GraphConfig.model_validate({**values, **changes})
    registry.register(model_id, replacement)
    return replacement
