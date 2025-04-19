from langchain_core.callbacks.base import Callbacks
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class GraphConfig(BaseModel):
    graph: CompiledStateGraph
    streamable_node_names: list[str]
    runtime_callbacks: list[Callbacks] | None = None

    class Config:
        arbitrary_types_allowed = True


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
            ValueError: If the graph name is not found in the registry.
        """
        if name not in self.registry:
            raise ValueError(f"Graph '{name}' not found in registry.")
        return self.registry[name]
