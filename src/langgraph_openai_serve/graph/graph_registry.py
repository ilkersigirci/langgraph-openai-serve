import inspect
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Self

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from langgraph_openai_serve.graph.client_settings import (
    ClientSettings,
    validate_client_settings_model,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.interrupt.coordination import RunCoordinator
from langgraph_openai_serve.graph.request import GraphRequest

GraphResolver = (
    CompiledStateGraph
    | Callable[[], CompiledStateGraph | Awaitable[CompiledStateGraph]]
)
RequestToInput = Callable[[GraphRequest, list[BaseMessage]], Any | Awaitable[Any]]
ContextFactory = Callable[
    [GraphRequest, Any],
    Any | Awaitable[Any],
]
OutputToMessage = Callable[[Any], AIMessage | Awaitable[AIMessage]]
_INTERRUPT_CHECKPOINTER_METHODS = (
    "aget_tuple",
    "alist",
    "aput",
    "aput_writes",
    "adelete_thread",
)


def _addressable_model_id(value: str) -> str:
    if value in {".", ".."}:
        msg = "model id must be addressable"
        raise ValueError(msg)
    return value


ModelId = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^[^/]+$"),
    AfterValidator(_addressable_model_id),
]


class GraphConfigurationError(RuntimeError):
    """Raised when a registered graph cannot satisfy its declared config."""


class GraphNotFoundError(ValueError):
    """Raised when a requested graph is not registered."""


class GraphConfig(BaseModel):
    """Graph configuration."""

    graph: GraphResolver
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    features: frozenset[GraphFeature] = frozenset()
    client_settings: type[ClientSettings] | None = None
    server_tools: frozenset[Annotated[str, StringConstraints(min_length=1)]] = (
        frozenset()
    )
    runtime_callbacks: Callbacks = None
    request_to_input: RequestToInput | None = None
    context_factory: ContextFactory | None = None
    output_to_message: OutputToMessage | None = None
    run_coordinator: RunCoordinator | None = None

    @field_validator("client_settings")
    @classmethod
    def validate_client_settings(
        cls,
        value: type[ClientSettings] | None,
    ) -> type[ClientSettings] | None:
        """Validate a public settings model when its graph is registered."""
        return validate_client_settings_model(value) if value is not None else None

    @model_validator(mode="after")
    def validate_interrupt_configuration(self) -> Self:
        """Validate feature relationships that do not depend on a resolved graph."""
        interrupt_enabled = self.supports(GraphFeature.INTERRUPTS)
        if self.run_coordinator is not None and not interrupt_enabled:
            msg = "run_coordinator is only supported by interrupt-enabled graphs."
            raise ValueError(msg)
        if interrupt_enabled and self.run_coordinator is None:
            msg = "Interrupt-enabled graphs must configure a run_coordinator."
            raise ValueError(msg)
        return self

    def supports(self, feature: GraphFeature) -> bool:
        """Return whether this graph supports a feature."""
        return feature in self.features

    async def resolve_graph(self) -> CompiledStateGraph:
        """Get the graph instance, resolving callable graph factories."""
        if isinstance(self.graph, CompiledStateGraph):
            graph = self.graph
        else:
            graph = await _maybe_await(self.graph())
        return _validate_resolved_graph(graph, self)

    async def build_input(
        self,
        request: GraphRequest,
        messages: list[BaseMessage],
    ) -> Any:
        """Build the native graph input for a normalized request."""
        if self.request_to_input is None:
            return {"messages": messages}
        return await _maybe_await(self.request_to_input(request, messages))

    async def build_context(
        self,
        request: GraphRequest,
        graph: CompiledStateGraph,
    ) -> Any:
        """Build the LangGraph runtime context for a request."""
        settings = (
            self.client_settings.validate_request(request)
            if self.client_settings is not None
            else None
        )
        if self.context_factory is not None:
            context = await _maybe_await(self.context_factory(request, settings))
        else:
            context = settings

        if context is None:
            return None
        if graph.context_schema is None:
            msg = "A graph that produces runtime context must declare context_schema."
            raise GraphConfigurationError(msg)

        # Preserve server-owned context objects; LangGraph applies context_schema
        # coercion when it invokes the graph.
        return context

    async def render_output(self, output: Any) -> AIMessage:
        """Convert native graph output into the durable assistant message."""
        if self.output_to_message is not None:
            return await _maybe_await(self.output_to_message(output))

        messages = (
            output["messages"]
            if isinstance(output, Mapping)
            else getattr(output, "messages", None)
        )
        if messages is None:
            msg = "Graph output must expose a messages field."
            raise GraphConfigurationError(msg)
        if not messages:
            return AIMessage(content="")
        message = messages[-1]
        if not isinstance(message, AIMessage):
            msg = "The final graph message must be an AIMessage."
            raise GraphConfigurationError(msg)
        return message

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


async def _maybe_await(value: Any | Awaitable[Any]) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _validate_resolved_graph(graph: object, config: GraphConfig) -> CompiledStateGraph:
    """Validate requirements that can change with each factory result."""
    if not isinstance(graph, CompiledStateGraph):
        msg = "Graph factories must return a compiled LangGraph StateGraph."
        raise GraphConfigurationError(msg)

    if (
        config.client_settings is not None
        and config.context_factory is None
        and graph.context_schema is not config.client_settings
    ):
        msg = (
            "Graphs using client_settings directly must use that settings model "
            "as context_schema."
        )
        raise GraphConfigurationError(msg)

    if config.supports(GraphFeature.INTERRUPTS):
        checkpointer = graph.checkpointer
        if checkpointer is None or any(
            not _overrides_checkpointer_method(checkpointer, method_name)
            for method_name in _INTERRUPT_CHECKPOINTER_METHODS
        ):
            msg = (
                "Interrupt-enabled graphs must use a fully asynchronous "
                "checkpointer with thread deletion."
            )
            raise GraphConfigurationError(msg)

    return graph


def _overrides_checkpointer_method(
    checkpointer: object,
    method_name: str,
) -> bool:
    """Reject async methods inherited unchanged from the saver's base stubs."""
    implementation = getattr(type(checkpointer), method_name, None)
    base_implementation = getattr(BaseCheckpointSaver, method_name)
    return callable(implementation) and implementation is not base_implementation


_MODEL_ID_ADAPTER = TypeAdapter(ModelId)


def _validate_model_id(value: object) -> str:
    return _MODEL_ID_ADAPTER.validate_python(value, strict=True)


def _validate_graph_config(value: object) -> GraphConfig:
    if not isinstance(value, GraphConfig):
        msg = "Registry values must be GraphConfig instances."
        raise TypeError(msg)
    return value


class GraphRegistry:
    """Registry of graphs."""

    __slots__ = ("_entries", "_registry")

    def __init__(self, *, registry: Mapping[str, GraphConfig]) -> None:
        if not registry:
            msg = "GraphRegistry must contain at least one graph."
            raise ValueError(msg)

        entries = {
            _validate_model_id(model_id): _validate_graph_config(config)
            for model_id, config in registry.items()
        }
        self._entries = entries
        self._registry = MappingProxyType(entries)

    @property
    def registry(self) -> Mapping[str, GraphConfig]:
        """The read-only, insertion-ordered registry view."""
        return self._registry

    def register(self, model_id: str, config: GraphConfig) -> None:
        """Add or replace one graph through the validated registry boundary."""
        validated_model_id = _validate_model_id(model_id)
        validated_config = _validate_graph_config(config)
        self._entries[validated_model_id] = validated_config

    def get_graph_names(self) -> list[str]:
        """Get the names of all registered graphs."""
        return list(self.registry.keys())

    def get_graph(self, name: str) -> GraphConfig:
        """
        Get a graph by its name.

        Args:
            name: The name of the graph to retrieve.

        Returns:
            The graph configuration associated with the given name.

        Raises:
            GraphNotFoundError: If the graph name is not found in the registry.

        """
        try:
            return self.registry[name]
        except KeyError as exc:
            msg = f"Graph '{name}' not found in registry."
            raise GraphNotFoundError(msg) from exc
