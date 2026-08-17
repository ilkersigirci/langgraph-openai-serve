"""FastAPI application for LangGraph with OpenAI compatible API.

This module provides a demo FastAPI application that exposes example LangGraph
graphs through the OpenAI-compatible API.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe

from lgos_demo_api.checkpointer import postgres_runtime
from lgos_demo_api.graphs.advanced_mcp import advanced_mcp_graph_config
from lgos_demo_api.graphs.citations import citation_graph_config
from lgos_demo_api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from lgos_demo_api.graphs.custom_events import custom_event_showcase_graph_config
from lgos_demo_api.graphs.custom_io import custom_io_graph_config
from lgos_demo_api.graphs.interruptible import (
    create_interruptible_graph,
    create_interruptible_graph_config,
)
from lgos_demo_api.graphs.lgos_rag import lgos_rag_graph_config
from lgos_demo_api.graphs.simple import simple_graph_config
from lgos_demo_api.graphs.status_events import status_event_graph_config
from lgos_demo_api.loggers.setup import setup_logging
from lgos_demo_api.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    This function handles the startup and shutdown events for the application.

    Args:
        app: The FastAPI application.
    """
    logger.info("Starting DEMO LangGraph OpenAI compatible server")

    async with postgres_runtime(settings.POSTGRES_URI) as runtime:
        app.state.interruptible_graph = create_interruptible_graph(runtime.checkpointer)
        app.state.interruptible_run_coordinator = runtime.run_coordinator
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
        version="0.0.1",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        # Local browser demos may use arbitrary origins; deployments must replace
        # this wildcard with their trusted origins.
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    graph_registry = GraphRegistry(
        registry={
            "citation-events": citation_graph_config,
            "simple-graph": simple_graph_config,
            "lgos-rag": lgos_rag_graph_config,
            "custom-input-output-context": custom_io_graph_config,
            "advanced-mcp-tools": advanced_mcp_graph_config,
            "complex-subgraphs": create_complex_subgraphs_graph_config(),
            "custom-event-showcase": custom_event_showcase_graph_config,
            "status-events": status_event_graph_config,
            "interruptible-approval": create_interruptible_graph_config(
                lambda: app.state.interruptible_graph,
                app.state.interruptible_run_coordinator,
            ),
        }
    )

    graph_serve = LanggraphOpenaiServe(
        app=app,
        graphs=graph_registry,
    )

    graph_serve.bind_openai_api()

    return app


app = create_custom_app()


def main() -> None:
    """Run the demo API with development defaults."""
    import uvicorn

    uvicorn.run(
        "lgos_demo_api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,
    )


if __name__ == "__main__":
    main()
