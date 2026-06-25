import inspect
from typing import Any, Awaitable, Callable

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest

GraphResolver = (
    CompiledStateGraph
    | Callable[[], CompiledStateGraph | Awaitable[CompiledStateGraph]]
)
RequestToInput = Callable[
    [ChatCompletionRequest, list[BaseMessage]], Any | Awaitable[Any]
]
ContextFactory = Callable[[ChatCompletionRequest], Any | Awaitable[Any]]
OutputToText = Callable[[Any], str | Awaitable[str]]


class GraphConfigurationError(RuntimeError):
    """Raised when a registered graph cannot satisfy its declared config."""


class GraphNotFoundError(ValueError):
    """Raised when a requested graph is not registered."""


class GraphConfig(BaseModel):
    graph: GraphResolver
    streamable_node_names: list[str] = Field(default_factory=list)
    runtime_callbacks: Callbacks = None
    request_to_input: RequestToInput | None = None
    context_factory: ContextFactory | None = None
    output_to_text: OutputToText | None = None
    interrupts_enabled: bool = False

    async def resolve_graph(self) -> CompiledStateGraph:
        """Get the graph instance, resolving callable graph factories."""
        if isinstance(self.graph, CompiledStateGraph):
            graph = self.graph
        else:
            graph = await _maybe_await(self.graph())

        if self.interrupts_enabled and graph.checkpointer is None:
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

    async def build_context(self, request: ChatCompletionRequest) -> Any:
        """Build the LangGraph runtime context for a chat completion request."""
        if self.context_factory is None:
            return None
        return await _maybe_await(self.context_factory(request))

    async def render_output(self, output: Any) -> str:
        """Convert native graph output into assistant response text."""
        if self.output_to_text is None:
            messages = output["messages"]
            return messages[-1].content if messages else ""
        return await _maybe_await(self.output_to_text(output))

    model_config = ConfigDict(arbitrary_types_allowed=True)


async def _maybe_await(value: Any | Awaitable[Any]) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class GraphRegistry(BaseModel):
    registry: dict[str, GraphConfig]

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
