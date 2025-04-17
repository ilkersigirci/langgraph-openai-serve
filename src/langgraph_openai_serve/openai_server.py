"""LangGraph OpenAI API Serve.

This module provides a server class that connects LangGraph instances to an OpenAI-compatible API.
It allows users to register their LangGraph instances and expose them through a FastAPI application.

Examples:
    >>> from langgraph_openai_serve import LangchainOpenaiApiServe
    >>> from fastapi import FastAPI
    >>> from your_graphs import simple_graph_1, simple_graph_2
    >>>
    >>> app = FastAPI(title="LangGraph OpenAI API")
    >>> graph_serve = LangchainOpenaiApiServe(
    ...     app=app,
    ...     graphs={
    ...         "simple_graph_1": simple_graph_1,
    ...         "simple_graph_2": simple_graph_2
    ...     }
    ... )
    >>> graph_serve.bind_openai_chat_completion(prefix="/v1")
"""

import logging
from typing import Any

from fastapi import FastAPI

from langgraph_openai_serve.api.chat import views as chat_views
from langgraph_openai_serve.api.health import views as health_views
from langgraph_openai_serve.api.models import views as models_views
from langgraph_openai_serve.graph.runner import register_graphs
from langgraph_openai_serve.graph.simple_graph import app as simple_graph

logger = logging.getLogger(__name__)


class LangchainOpenaiApiServe:
    """Server class to connect LangGraph instances with an OpenAI-compatible API.

    This class serves as a bridge between LangGraph instances and an OpenAI-compatible API.
    It allows users to register their LangGraph instances and expose them through a FastAPI application.

    Attributes:
        app: The FastAPI application to attach routers to.
        graphs: A dictionary mapping graph names to LangGraph instances.
    """

    def __init__(
        self,
        app: FastAPI | None = None,
        graphs: dict[str, Any] | None = None,
        configure_cors: bool = False,
    ):
        """Initialize the server with a FastAPI app (optional) and LangGraph instances.(optional)

        Args:
            app: The FastAPI application to attach routers to. If None, a new FastAPI app will be created.
            graphs: A dictionary mapping graph names to LangGraph instances. If None, a default simple graph will be used.
            configure_cors: Optional; Whether to configure CORS for the FastAPI application.
        """
        self.app = app
        self.graphs = graphs

        if app is None:
            app = FastAPI(
                title="LangGraph OpenAI Compatible API",
                description="An OpenAI-compatible API for LangGraph",
                version="0.0.1",
            )

        if graphs is None:
            logger.info("Graphs not provided, using default simple graph")
            graphs = {
                "simple-graph": simple_graph,
            }

        self.app = app
        self.graphs = graphs

        # Configure CORS if requested
        if configure_cors:
            self._configure_cors()

        # Register the graphs with the graph runner (now uses global variable)
        register_graphs(self.graphs)

        logger.info(f"Initialized LangchainOpenaiApiServe with {len(graphs)} graphs")
        logger.info(f"Available graphs: {', '.join(graphs.keys())}")

    def bind_openai_chat_completion(self, prefix: str = "/v1"):
        """Bind OpenAI-compatible chat completion endpoints to the FastAPI app.

        Args:
            prefix: Optional; The URL prefix for the OpenAI-compatible endpoints. Defaults to "/v1".
        """
        # Include routers with the specified prefix
        self.app.include_router(chat_views.router, prefix=prefix)
        self.app.include_router(health_views.router, prefix=prefix)
        self.app.include_router(models_views.router, prefix=prefix)

        logger.info(f"Bound OpenAI chat completion endpoints with prefix: {prefix}")

        return self
