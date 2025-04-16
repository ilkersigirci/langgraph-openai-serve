"""FastAPI application for LangGraph with OpenAI compatible API.

This module provides a FastAPI application that implements an OpenAI-compatible
API for LangGraph, allowing clients to interact with LangGraph models using
the same interface as OpenAI's API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langgraph_openai_serve.loggers.setup import setup_logging
from langgraph_openai_serve.routers import chat, health, models
from langgraph_openai_serve.utils.general import check_env_vars

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    This function handles the startup and shutdown events for the application.

    Args:
        app: The FastAPI application.
    """
    # Startup
    logger.info("Starting LangGraph OpenAI compatible server")
    # Additional startup logic here (e.g., loading models)

    yield

    # Shutdown
    logger.info("Shutting down LangGraph OpenAI compatible server")
    # Additional cleanup logic here


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI application.
    """
    # Set up logging
    setup_logging()

    # Check required environment variables
    # If you have required env vars, add them here
    check_env_vars([])

    # Create the FastAPI app
    app = FastAPI(
        title="LangGraph OpenAI Compatible API",
        description="An OpenAI-compatible API for LangGraph",
        version="0.0.1",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "langgraph_openai_serve.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,  # Use our custom logger
    )
