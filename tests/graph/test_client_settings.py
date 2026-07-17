from dataclasses import dataclass
from datetime import date
from typing import cast

import pytest
from langgraph.graph import StateGraph
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
)

from langgraph_openai_serve import ClientSettings, GraphConfig
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.graph.client_settings import (
    RUNTIME_SETTINGS_METADATA_KEY,
    ClientSettingsValidationError,
)
from langgraph_openai_serve.graph.graph_registry import GraphConfigurationError
from tests.graph.support.schemas import MessageState

pytestmark = pytest.mark.anyio


class PublicSettings(ClientSettings):
    enabled: bool = True
    day: date = date(2026, 7, 17)


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
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Hello"}],
        metadata=(
            {RUNTIME_SETTINGS_METADATA_KEY: settings} if settings is not None else None
        ),
    )


def test_client_settings_own_the_public_contract_and_defaults() -> None:
    graph_config = GraphConfig(
        graph=make_context_graph(PublicSettings),
        client_settings=PublicSettings,
    )

    assert graph_config.client_settings is PublicSettings
    assert PublicSettings.defaults() == PublicSettings()
    assert PublicSettings.model_json_schema()["additionalProperties"] is False


def test_client_settings_require_a_complete_default(message_graph) -> None:
    class RequiredSettings(ClientSettings):
        required: int

    with pytest.raises(ValidationError, match="Field required"):
        GraphConfig(graph=message_graph, client_settings=RequiredSettings)


def test_client_settings_deeply_validate_nested_defaults(message_graph) -> None:
    class NestedValue(BaseModel):
        count: int

    invalid = NestedValue.model_construct(count=cast(int, "one"))

    class InvalidSettings(ClientSettings):
        nested: NestedValue = invalid

    with pytest.raises(ValidationError, match="serialized value"):
        GraphConfig(graph=message_graph, client_settings=InvalidSettings)


def test_client_settings_reject_non_finite_defaults(message_graph) -> None:
    class InvalidSettings(ClientSettings):
        number: float = float("inf")

    with pytest.raises(ValidationError, match="finite number"):
        GraphConfig(graph=message_graph, client_settings=InvalidSettings)


def test_client_settings_reject_non_finite_schema_extras(message_graph) -> None:
    class InvalidSettings(ClientSettings):
        model_config = ConfigDict(
            json_schema_extra={"extension": {"number": float("nan")}}
        )

        enabled: bool = True

    with pytest.raises(ValidationError, match="finite number"):
        GraphConfig(graph=message_graph, client_settings=InvalidSettings)


def test_default_factory_is_evaluated_once_for_discovery_and_requests() -> None:
    calls = 0

    def next_default() -> int:
        nonlocal calls
        calls += 1
        return calls

    class FactorySettings(ClientSettings):
        value: int = Field(default_factory=next_default)

    GraphConfig(
        graph=make_context_graph(FactorySettings),
        client_settings=FactorySettings,
    )

    assert FactorySettings.defaults().value == 1
    assert FactorySettings.validate_request(make_request()).value == 1
    assert calls == 1


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
    with pytest.raises(ClientSettingsValidationError) as exc_info:
        PublicSettings.validate_request(make_request(settings='{"enabled":"false"}'))

    assert "Input should be a valid boolean" in str(exc_info.value)
    assert exc_info.value.param == f"metadata.{RUNTIME_SETTINGS_METADATA_KEY}"


def test_runtime_settings_must_be_a_json_object() -> None:
    with pytest.raises(ClientSettingsValidationError) as exc_info:
        PublicSettings.validate_request(make_request(settings="[]"))

    assert "Input should be an object" in str(exc_info.value)
    assert exc_info.value.param == f"metadata.{RUNTIME_SETTINGS_METADATA_KEY}"


def test_runtime_settings_reject_non_finite_json_values() -> None:
    class JsonSettings(ClientSettings):
        payload: JsonValue = Field(default_factory=dict)

    with pytest.raises(ClientSettingsValidationError, match="finite number"):
        JsonSettings.validate_request(
            make_request(settings='{"payload":{"number":Infinity}}')
        )


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
        client_settings=PublicSettings,
        context_factory=context_factory,
    )
    request = make_request(settings='{"enabled":false}')
    request.user = "alice"

    context = await graph_config.build_context(request, graph)

    assert isinstance(received_settings, PublicSettings)
    assert context == {
        "settings": PublicSettings(enabled=False),
        "user_id": "alice",
    }


async def test_direct_settings_require_the_same_graph_context_schema(
    message_graph,
) -> None:
    graph_config = GraphConfig(graph=message_graph, client_settings=PublicSettings)

    with pytest.raises(GraphConfigurationError, match="must use that settings model"):
        await graph_config.resolve_graph()


async def test_lazy_graph_non_null_context_requires_schema(
    message_graph,
) -> None:
    graph_config = GraphConfig(
        graph=lambda: message_graph,
        context_factory=lambda _request, _settings: {"user_id": "alice"},
    )

    graph = await graph_config.resolve_graph()

    with pytest.raises(GraphConfigurationError, match="declare context_schema"):
        await graph_config.build_context(make_request(), graph)


def test_client_settings_cannot_relax_required_model_config(message_graph) -> None:
    class RelaxedSettings(ClientSettings):
        model_config = ConfigDict(extra="allow", frozen=False)

        enabled: bool = True

    with pytest.raises(ValidationError, match="preserve the inherited model config"):
        GraphConfig(graph=message_graph, client_settings=RelaxedSettings)


def test_client_settings_fields_cannot_be_excluded_from_defaults(message_graph) -> None:
    class ExcludedSettings(ClientSettings):
        hidden: str = Field(default="hidden", exclude=True)

    with pytest.raises(ValidationError, match="cannot be excluded from defaults"):
        GraphConfig(graph=message_graph, client_settings=ExcludedSettings)
