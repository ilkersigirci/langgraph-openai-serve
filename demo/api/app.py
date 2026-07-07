"""FastAPI application for LangGraph with OpenAI compatible API.

This module provides a default FastAPI application that implements an OpenAI-compatible
API for LangGraph, allowing clients to interact with LangGraph models using
the same interface as OpenAI's API.

For more flexibility and control, users can create their own applications
using the LangchainOpenaiApiServe class directly.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from demo.api.graphs.advanced_mcp import advanced_mcp_graph
from demo.api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from demo.api.graphs.custom_io import custom_io_graph_config
from demo.api.graphs.interruptible import create_interruptible_graph
from demo.api.graphs.simple import simple_graph
from demo.api.loggers.setup import setup_logging
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe

logger = logging.getLogger(__name__)


def create_default_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        A default FastAPI application.
    """

    # Set up logging
    setup_logging()

    graph_serve = LangchainOpenaiApiServe()

    # Bind the OpenAI-compatible endpoints at settings.OPENAI_API_PREFIX.
    graph_serve.bind_openai_chat_completion()

    return graph_serve.app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    This function handles the startup and shutdown events for the application.

    Args:
        app: The FastAPI application.
    """
    logger.info("Starting DEMO LangGraph OpenAI compatible server")

    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app.state.interruptible_graph = create_interruptible_graph(checkpointer)
        yield

    logger.info("Shutting down DEMO LangGraph OpenAI compatible server")


def create_custom_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI application.
    """

    setup_logging()

    app = FastAPI(
        title="Demo",
        description="Demo LangGraph OpenAI-compatible API",
        version="0.0.1",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    simple_graph_with_history = simple_graph.with_config(
        configurable={"use_history": True},
    )

    simple_graph_no_history = simple_graph.with_config(
        configurable={"use_history": False},
    )

    graph_registry = GraphRegistry(
        registry={
            "simple-graph-with-history": GraphConfig(
                graph=simple_graph_with_history, streamable_node_names=["generate"]
            ),
            "simple-graph-no-history": GraphConfig(
                graph=simple_graph_no_history, streamable_node_names=["generate"]
            ),
            "custom-input-output-context": custom_io_graph_config,
            "advanced-mcp-tools": GraphConfig(graph=advanced_mcp_graph),
            "complex-subgraphs": create_complex_subgraphs_graph_config(),
            "interruptible-approval": GraphConfig(
                graph=lambda: app.state.interruptible_graph,
                request_to_input=lambda request, messages: {
                    "request": messages[-1].content or ""
                },
                output_to_text=lambda output: output["response"],
                interrupts_enabled=True,
            ),
        }
    )

    graph_serve = LangchainOpenaiApiServe(
        app=app,
        graphs=graph_registry,
    )

    # Bind the OpenAI-compatible endpoints at settings.OPENAI_API_PREFIX.
    graph_serve.bind_openai_chat_completion()

    return graph_serve.app


# app = create_default_app()
app = create_custom_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,
    )
