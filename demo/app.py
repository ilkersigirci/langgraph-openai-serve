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

from demo.loggers.setup import setup_logging
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe
from langgraph_openai_serve.graph.simple_graph import app as simple_graph

logger = logging.getLogger(__name__)


def create_default_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        A default FastAPI application.
    """

    # Set up logging
    setup_logging()

    graph_serve = LangchainOpenaiApiServe()

    # Bind the OpenAI-compatible endpoints
    graph_serve.bind_openai_chat_completion(prefix="/v1")

    return graph_serve.app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    This function handles the startup and shutdown events for the application.

    Args:
        app: The FastAPI application.
    """
    # Startup
    logger.info("Starting DEMO LangGraph OpenAI compatible server")
    # Additional startup logic here (e.g., loading models)

    yield

    # Shutdown
    logger.info("Shutting down DEMO LangGraph OpenAI compatible server")
    # Additional cleanup logic here


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
        }
    )

    graph_serve = LangchainOpenaiApiServe(
        app=app,
        graphs=graph_registry,
    )

    # Bind the OpenAI-compatible endpoints
    graph_serve.bind_openai_chat_completion(prefix="/v1")

    return graph_serve.app


# app = create_default_app()
app = create_custom_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,
    )
