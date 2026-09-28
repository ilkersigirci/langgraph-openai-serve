from dataclasses import dataclass
from datetime import date

import pytest
from langgraph.graph import StateGraph
from pydantic import Field, ValidationError

from langgraph_openai_serve import ClientSettings, GraphConfig, GraphRequest
from langgraph_openai_serve.core.errors import GraphError, InvalidRequestError
from langgraph_openai_serve.protocol import JSON_SCHEMA_DIALECT, SETTINGS_METADATA_KEY
from tests.graph.support.schemas import MessageState


class PublicSettings(ClientSettings):
    enabled: bool = True
    day: date = date(2026, 7, 17)


@pytest.mark.parametrize("kwargs", [{}, {"description": "   "}])
def test_graph_description_is_required(message_graph, kwargs) -> None:
    with pytest.raises(ValidationError, match="description"):
        GraphConfig(graph=message_graph, **kwargs)


def test_graph_description_is_trimmed(message_graph) -> None:
    graph_config = GraphConfig(graph=message_graph, description="  DUMMY  ")

    assert graph_config.description == "DUMMY"


def make_context_graph(context_schema):
    return (
        StateGraph(MessageState, context_schema=context_schema)
        .add_node("echo", lambda state: state)
        .set_entry_point("echo")
        .set_finish_point("echo")
        .compile()
    )


def make_request(
    *,
    settings: str | None = None,
    user: str | None = None,
) -> GraphRequest:
    return GraphRequest(
        model="test",
        metadata={SETTINGS_METADATA_KEY: settings} if settings else {},
        user=user,
        tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
    )


def test_client_settings_own_the_public_contract_and_defaults() -> None:
    graph_config = GraphConfig(
        graph=make_context_graph(PublicSettings),
        description="DUMMY",
        client_settings=PublicSettings,
    )

    assert graph_config.client_settings is PublicSettings
    assert PublicSettings.default_values() == {
        "enabled": True,
        "day": "2026-07-17",
    }
    schema = PublicSettings.json_schema()
    assert schema["$schema"] == JSON_SCHEMA_DIALECT
    assert schema["additionalProperties"] is False


def test_aliased_settings_are_advertised_and_accepted_by_alias() -> None:
    class AliasedSettings(ClientSettings):
        top_k: int = Field(default=3, alias="topK")

    assert list(AliasedSettings.json_schema()["properties"]) == ["topK"]
    assert AliasedSettings.default_values() == {"topK": 3}
    settings = AliasedSettings.validate_request(make_request(settings='{"topK":5}'))
    assert settings == AliasedSettings(topK=5)


def test_client_settings_require_a_complete_default(message_graph) -> None:
    class RequiredSettings(ClientSettings):
        required: int

    with pytest.raises(ValidationError, match="Field required"):
        GraphConfig(
            graph=message_graph,
            description="DUMMY",
            client_settings=RequiredSettings,
        )


def test_client_settings_reject_non_finite_defaults(message_graph) -> None:
    class InvalidSettings(ClientSettings):
        number: float = float("inf")

    with pytest.raises(ValidationError, match="finite number"):
        GraphConfig(
            graph=message_graph,
            description="DUMMY",
            client_settings=InvalidSettings,
        )


def test_request_validation_uses_strict_json_mode() -> None:
    settings = PublicSettings.validate_request(
        make_request(
            settings='{"enabled":false,"day":"2026-07-18"}',
        )
    )

    assert settings == PublicSettings(
        enabled=False,
        day=date(2026, 7, 18),
    )


def test_request_validation_does_not_coerce_json_values() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        PublicSettings.validate_request(make_request(settings='{"enabled":"false"}'))

    assert "Input should be a valid boolean" in str(exc_info.value)
    assert exc_info.value.param == f"metadata.{SETTINGS_METADATA_KEY}"


def test_runtime_settings_must_be_a_json_object() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        PublicSettings.validate_request(make_request(settings="[]"))

    assert "Input should be an object" in str(exc_info.value)
    assert exc_info.value.param == f"metadata.{SETTINGS_METADATA_KEY}"


@dataclass
class RuntimeContext:
    settings: PublicSettings
    user_id: str


async def test_context_factory_composes_public_and_server_context() -> None:
    graph = make_context_graph(RuntimeContext)
    received_settings = None

    def context_factory(request, settings):
        nonlocal received_settings
        received_settings = settings
        return {"settings": settings, "user_id": request.user}

    graph_config = GraphConfig(
        graph=graph,
        description="DUMMY",
        client_settings=PublicSettings,
        context_factory=context_factory,
    )
    request = make_request(settings='{"enabled":false}', user="alice")

    context = await graph_config.build_context(request, graph)

    assert isinstance(received_settings, PublicSettings)
    assert context == {
        "settings": PublicSettings(enabled=False),
        "user_id": "alice",
    }


async def test_direct_settings_require_the_same_graph_context_schema(
    message_graph,
) -> None:
    graph_config = GraphConfig(
        graph=message_graph,
        description="DUMMY",
        client_settings=PublicSettings,
    )

    with pytest.raises(GraphError, match="must use that settings model"):
        await graph_config.resolve_graph()


async def test_lazy_graph_non_null_context_requires_schema(
    message_graph,
) -> None:
    graph_config = GraphConfig(
        graph=lambda: message_graph,
        description="DUMMY",
        context_factory=lambda _request, _settings: {"user_id": "alice"},
    )

    graph = await graph_config.resolve_graph()

    with pytest.raises(GraphError, match="declare context_schema"):
        await graph_config.build_context(make_request(), graph)
