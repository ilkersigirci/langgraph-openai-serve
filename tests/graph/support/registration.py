"""Graph-registration helpers shared by API behavior tests."""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from langgraph_openai_serve import GraphConfig, GraphRegistry, GraphRequest


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
    registry.graphs[model_id] = replacement
    return replacement


@dataclass
class GraphInputs:
    """The request and messages that each run of a model received."""

    requests: list[GraphRequest] = field(default_factory=list)
    messages: list[list[BaseMessage]] = field(default_factory=list)


def capture_graph_inputs(
    registry: GraphRegistry,
    model_id: str = "test",
    **changes: object,
) -> GraphInputs:
    """Record what each run of ``model_id`` receives, then run it on its messages."""
    inputs = GraphInputs()

    def capture(
        request: GraphRequest,
        messages: list[BaseMessage],
    ) -> dict[str, list[BaseMessage]]:
        inputs.requests.append(request)
        inputs.messages.append(messages)
        return {"messages": messages}

    replace_graph_config(registry, model_id, request_to_input=capture, **changes)
    return inputs
