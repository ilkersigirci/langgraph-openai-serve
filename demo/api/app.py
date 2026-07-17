"""FastAPI application for LangGraph with OpenAI compatible API.

This module provides a demo FastAPI application that exposes example LangGraph
graphs through the OpenAI-compatible API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from demo.api.checkpointer import postgres_checkpointer
from demo.api.graphs.advanced_mcp import advanced_mcp_graph
from demo.api.graphs.citations import citation_graph
from demo.api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from demo.api.graphs.custom_events import custom_event_showcase_graph_config
from demo.api.graphs.custom_io import custom_io_graph_config
from demo.api.graphs.interruptible import create_interruptible_graph
from demo.api.graphs.lgos_rag import lgos_rag
from demo.api.graphs.simple import SimpleContext, simple_graph
from demo.api.loggers.setup import setup_logging
from demo.api.settings import settings
from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    LanggraphOpenaiServe,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    This function handles the startup and shutdown events for the application.

    Args:
        app: The FastAPI application.
    """
    logger.info("Starting DEMO LangGraph OpenAI compatible server")

    async with postgres_checkpointer(settings.POSTGRES_URI) as checkpointer:
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

    graph_registry = GraphRegistry(
        registry={
            "citation-events": GraphConfig(
                graph=citation_graph,
                streamable_node_names=["answer_with_citation"],
            ),
            "simple-graph": GraphConfig(
                graph=simple_graph,
                streamable_node_names=["generate"],
                client_settings=SimpleContext,
            ),
            "lgos-rag": GraphConfig(
                graph=lgos_rag,
                streamable_node_names=[
                    "generate_query_or_respond",
                    "generate_answer",
                    "answer_no_results",
                ],
            ),
            "custom-input-output-context": custom_io_graph_config,
            "advanced-mcp-tools": GraphConfig(graph=advanced_mcp_graph),
            "complex-subgraphs": create_complex_subgraphs_graph_config(),
            "custom-event-showcase": custom_event_showcase_graph_config,
            "interruptible-approval": GraphConfig(
                graph=lambda: app.state.interruptible_graph,
                request_to_input=lambda request, messages: {
                    "request": messages[-1].content or ""
                },
                output_to_text=lambda output: output["response"],
                features={GraphFeature.INTERRUPTS},
            ),
        }
    )

    graph_serve = LanggraphOpenaiServe(
        app=app,
        graphs=graph_registry,
    )

    # Bind the OpenAI-compatible endpoints at settings.OPENAI_API_PREFIX.
    graph_serve.bind_openai_api()

    return app


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
