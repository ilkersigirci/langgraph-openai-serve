import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Awaitable, Callable

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.graph.client_config import (
    ClientSettings,
    validate_client_settings_model,
)
from langgraph_openai_serve.graph.features import GraphFeature

GraphResolver = (
    CompiledStateGraph
    | Callable[[], CompiledStateGraph | Awaitable[CompiledStateGraph]]
)
RequestToInput = Callable[
    [ChatCompletionRequest, list[BaseMessage]], Any | Awaitable[Any]
]
ContextFactory = Callable[
    [ChatCompletionRequest, Any],
    Any | Awaitable[Any],
]
OutputToText = Callable[[Any], str | Awaitable[str]]


def _addressable_model_id(value: str) -> str:
    if value in {".", ".."}:
        raise ValueError("model id must be addressable")
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
    graph: GraphResolver
    streamable_node_names: list[str] = Field(default_factory=list)
    features: set[GraphFeature] = Field(default_factory=set)
    client_config: type[ClientSettings] | None = None
    runtime_callbacks: Callbacks = None
    request_to_input: RequestToInput | None = None
    context_factory: ContextFactory | None = None
    output_to_text: OutputToText | None = None

    @field_validator("client_config")
    @classmethod
    def validate_client_config(
        cls,
        value: type[ClientSettings] | None,
    ) -> type[ClientSettings] | None:
        """Validate a public settings model when its graph is registered."""
        return validate_client_settings_model(value) if value is not None else None

    def supports(self, feature: GraphFeature) -> bool:
        """Return whether this graph supports a feature."""
        return feature in self.features

    async def resolve_graph(self) -> CompiledStateGraph:
        """Get the graph instance, resolving callable graph factories."""
        if isinstance(self.graph, CompiledStateGraph):
            graph = self.graph
        else:
            graph = await _maybe_await(self.graph())

        if self.supports(GraphFeature.INTERRUPTS) and graph.checkpointer is None:
            raise GraphConfigurationError(
                "Interrupt-enabled graphs must be compiled with a checkpointer."
            )

        return graph

    async def build_input(
        self,
        request: ChatCompletionRequest,
        messages: list[BaseMessage],
    ) -> Any:
        """Build the native graph input for a chat completion request."""
        if self.request_to_input is None:
            return {"messages": messages}
        return await _maybe_await(self.request_to_input(request, messages))

    async def build_context(
        self,
        request: ChatCompletionRequest,
        graph: CompiledStateGraph,
    ) -> Any:
        """Build and validate the LangGraph runtime context for a request."""
        settings = (
            self.client_config.validate_request(request)
            if self.client_config is not None
            else None
        )
        if self.context_factory is not None:
            context = await _maybe_await(self.context_factory(request, settings))
        else:
            context = settings

        if context is None or graph.context_schema is None:
            return context

        try:
            return TypeAdapter(graph.context_schema).validate_python(context)
        except ValidationError as exc:
            raise GraphConfigurationError(
                "The configured context does not match the graph's context schema: "
                f"{exc.errors()[0]['msg']}"
            ) from exc

    async def render_output(self, output: Any) -> str:
        """Convert native graph output into assistant response text."""
        if self.output_to_text is None:
            messages = output["messages"]
            return messages[-1].content if messages else ""
        return await _maybe_await(self.output_to_text(output))

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


async def _maybe_await(value: Any | Awaitable[Any]) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _freeze_registry(
    value: Mapping[ModelId, GraphConfig],
) -> Mapping[ModelId, GraphConfig]:
    return MappingProxyType(dict(value))


_RegistryEntries = Annotated[
    Mapping[ModelId, GraphConfig],
    Field(min_length=1),
    AfterValidator(_freeze_registry),
    PlainSerializer(dict, return_type=dict),
]


class GraphRegistry(BaseModel):
    registry: _RegistryEntries

    model_config = ConfigDict(validate_assignment=True)

    def register(self, model_id: str, config: GraphConfig) -> None:
        """Add or replace one graph through the validated registry boundary."""
        self.registry = {**self.registry, model_id: config}

    def get_graph_names(self) -> list[str]:
        """Get the names of all registered graphs."""
        return list(self.registry.keys())

    def get_graph(self, name: str) -> GraphConfig:
        """Get a graph by its name.

        Args:
            name: The name of the graph to retrieve.

        Returns:
            The graph configuration associated with the given name.

        Raises:
            GraphNotFoundError: If the graph name is not found in the registry.
        """
        if name not in self.registry:
            raise GraphNotFoundError(f"Graph '{name}' not found in registry.")
        return self.registry[name]
