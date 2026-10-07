"""The demo catalog that ``lgos serve`` and ``lgos worker`` run."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.server import ServerResources

from lgos_demo_api.graphs.advanced_graph import (
    create_advanced_graph_config,
    open_advanced_graph,
)
from lgos_demo_api.graphs.background_interrupt import (
    create_background_interrupt_graph,
    create_background_interrupt_graph_config,
)
from lgos_demo_api.graphs.background_mock import background_mock_graph_config
from lgos_demo_api.graphs.citations import citation_graph_config
from lgos_demo_api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from lgos_demo_api.graphs.custom_io import custom_io_graph_config
from lgos_demo_api.graphs.file_input import file_input_graph_config
from lgos_demo_api.graphs.interruptible import (
    create_interruptible_graph,
    create_interruptible_graph_config,
)
from lgos_demo_api.graphs.lgos_rag import lgos_rag_graph_config
from lgos_demo_api.graphs.mcp_mock import mcp_mock_graph_config
from lgos_demo_api.graphs.mcp_postgres import mcp_postgres_graph_config
from lgos_demo_api.graphs.multi_node_streaming import (
    multi_node_streaming_graph_config,
)
from lgos_demo_api.graphs.persistent_plot_agent import (
    create_persistent_plot_agent,
    create_persistent_plot_agent_config,
)
from lgos_demo_api.graphs.response_outcomes import response_outcome_graph_config
from lgos_demo_api.graphs.server_tool import server_tool_graph_config
from lgos_demo_api.graphs.simple import simple_graph_config
from lgos_demo_api.graphs.simple_external_tools import (
    simple_external_tools_graph_config,
)
from lgos_demo_api.graphs.status_events import status_event_graph_config
from lgos_demo_api.graphs.streaming_long_mock import (
    streaming_long_mock_graph_config,
)


@asynccontextmanager
async def open_registry(
    resources: ServerResources,
) -> AsyncGenerator[GraphRegistry, None]:
    """
    Open every demo model over the server's persistence.

    The advanced graph owns HTTP clients for the process lifetime, so the
    catalog is a context manager rather than a plain registry.

    Yields:
        The demo models keyed by OpenAI model ID.

    """
    interruptible_graph = create_interruptible_graph(resources.checkpointer)
    background_interrupt_graph = create_background_interrupt_graph(
        resources.checkpointer
    )
    persistent_plot_agent = create_persistent_plot_agent(resources.store)
    async with open_advanced_graph(
        resources.checkpointer, resources.store
    ) as advanced_graph:
        yield GraphRegistry(
            graphs={
                "advanced-graph": create_advanced_graph_config(advanced_graph),
                "background-mock": background_mock_graph_config,
                "background-interrupt": create_background_interrupt_graph_config(
                    background_interrupt_graph,
                ),
                "citation-events": citation_graph_config,
                "file-input": file_input_graph_config,
                "simple-graph": simple_graph_config,
                "server-tool": server_tool_graph_config,
                "lgos-rag": lgos_rag_graph_config,
                "custom-input-output-context": custom_io_graph_config,
                "mcp-mock": mcp_mock_graph_config,
                "mcp-postgres": mcp_postgres_graph_config,
                "complex-subgraphs": create_complex_subgraphs_graph_config(),
                "multi-node-streaming": multi_node_streaming_graph_config,
                "streaming-long-mock": streaming_long_mock_graph_config,
                "status-events": status_event_graph_config,
                "response-outcomes": response_outcome_graph_config,
                "persistent-plot-agent": create_persistent_plot_agent_config(
                    persistent_plot_agent,
                ),
                "simple-graph-external-tools": simple_external_tools_graph_config,
                "interruptible-approval": create_interruptible_graph_config(
                    interruptible_graph,
                ),
            },
            run_coordinator=resources.run_coordinator,
        )
