import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Self

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from langgraph_openai_serve.core.errors import (
    GraphError,
    InvalidRequestError,
)
from langgraph_openai_serve.graph.client_settings import ClientSettings
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.interrupt import RunCoordinator
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
    "aput",
    "aput_writes",
    "adelete_thread",
)


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

    @field_validator("client_settings")
    @classmethod
    def validate_client_settings(
        cls,
        value: type[ClientSettings] | None,
    ) -> type[ClientSettings] | None:
        """Fail at registration when a settings model cannot be advertised."""
        if value is not None:
            value.json_schema()
            value.default_values()
        return value

    @model_validator(mode="after")
    def validate_compiled_graph(self) -> Self:
        """Fail at registration when a directly supplied graph cannot be served."""
        if isinstance(self.graph, CompiledStateGraph):
            _validate_resolved_graph(self.graph, self)
        return self

    def supports(self, feature: GraphFeature) -> bool:
        """Return whether this graph supports a feature."""
        return feature in self.features

    async def resolve_graph(self) -> CompiledStateGraph:
        """Get the graph instance, resolving callable graph factories."""
        if isinstance(self.graph, CompiledStateGraph):
            return self.graph
        return _validate_resolved_graph(await _maybe_await(self.graph()), self)

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
            raise GraphError(msg)

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
            raise GraphError(msg)
        if not messages:
            return AIMessage(content="")
        message = messages[-1]
        if not isinstance(message, AIMessage):
            msg = "The final graph message must be an AIMessage."
            raise GraphError(msg)
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
    """Validate a supplied graph once, and a factory's result on every call."""
    if not isinstance(graph, CompiledStateGraph):
        msg = "Graph factories must return a compiled LangGraph StateGraph."
        raise GraphError(msg)

    # LGOS reports only interrupt() pauses; a static breakpoint would end the run
    # early and render its partial state as the final answer.
    if graph.interrupt_before_nodes or graph.interrupt_after_nodes:
        msg = (
            "Graphs must pause with interrupt(); compile them without "
            "interrupt_before or interrupt_after."
        )
        raise GraphError(msg)

    if (
        config.client_settings is not None
        and config.context_factory is None
        and graph.context_schema is not config.client_settings
    ):
        msg = (
            "Graphs using client_settings directly must use that settings model "
            "as context_schema."
        )
        raise GraphError(msg)

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
            raise GraphError(msg)
    elif isinstance(graph.checkpointer, BaseCheckpointSaver):
        # LGOS supplies a checkpoint thread only to interrupt runs; clients send
        # the full conversation with every other request.
        msg = "Only interrupt-enabled graphs may be compiled with a checkpointer."
        raise GraphError(msg)

    return graph


def _overrides_checkpointer_method(
    checkpointer: object,
    method_name: str,
) -> bool:
    """Reject async methods inherited unchanged from the saver's base stubs."""
    implementation = getattr(type(checkpointer), method_name, None)
    base_implementation = getattr(BaseCheckpointSaver, method_name)
    return callable(implementation) and implementation is not base_implementation


@dataclass
class GraphRegistry:
    """
    The graphs served as OpenAI models, keyed by model ID.

    ``run_coordinator`` leases the interrupt runs of every interrupt-enabled
    graph; use a shared coordinator when several processes serve these graphs.
    """

    graphs: dict[str, GraphConfig]
    run_coordinator: RunCoordinator | None = None

    def __post_init__(self) -> None:
        """Fail at startup for a registry the OpenAI routes cannot serve."""
        if not self.graphs:
            msg = "GraphRegistry must contain at least one graph."
            raise ValueError(msg)
        for model_id in self.graphs:
            # Each model must be addressable as GET /models/{model}.
            if not model_id or "/" in model_id or model_id in {".", ".."}:
                msg = f"Model ID {model_id!r} is not addressable."
                raise ValueError(msg)
        if self.run_coordinator is None and any(
            config.supports(GraphFeature.INTERRUPTS) for config in self.graphs.values()
        ):
            msg = "Interrupt-enabled graphs need a GraphRegistry run_coordinator."
            raise ValueError(msg)

    def get_graph(self, model: str) -> GraphConfig:
        """Return the graph served as ``model``."""
        try:
            return self.graphs[model]
        except KeyError as exc:
            msg = f"The model '{model}' does not exist."
            raise InvalidRequestError(
                msg,
                param="model",
                code="model_not_found",
                status_code=404,
            ) from exc
