"""LangGraph OpenAI API Serve.

This module provides a server class that connects LangGraph instances to an
OpenAI-compatible API. It allows users to register their LangGraph instances
and expose them through a mounted FastAPI sub-application.

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
    >>> graph_serve.bind_openai_chat_completion()
"""

import logging

from fastapi import FastAPI

from langgraph_openai_serve.api.chat import views as chat_views
from langgraph_openai_serve.api.health import views as health_views
from langgraph_openai_serve.api.models import views as models_views
from langgraph_openai_serve.core.errors import configure_openai_error_handlers
from langgraph_openai_serve.core.settings import normalize_openai_api_prefix, settings
from langgraph_openai_serve.core.version import get_version
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.simple_graph import app as simple_graph

logger = logging.getLogger(__name__)


class LangchainOpenaiApiServe:
    """Server class to connect LangGraph instances with an OpenAI-compatible API.

    This class serves as a bridge between LangGraph instances and an OpenAI-compatible API.
    It allows users to register their LangGraph instances and expose them
    through an OpenAI-compatible sub-application mounted on a FastAPI host app.

    Attributes:
        app: The host FastAPI application to mount the OpenAI API on.
        graphs: A GraphRegistry instance containing the graphs to serve.
    """

    def __init__(
        self,
        app: FastAPI | None = None,
        graphs: GraphRegistry | None = None,
        configure_cors: bool = False,
    ):
        """Initialize the server with a FastAPI app (optional) and a GraphRegistry instance (optional).

        Args:
            app: The host FastAPI application to mount the OpenAI API on. If None,
                a new FastAPI app will be created.
            graphs: A GraphRegistry instance containing the graphs to serve.
                    If None, a default simple graph will be used.
            configure_cors: Optional; Whether to configure CORS for the FastAPI application.
        """
        self.app = app

        if app is None:
            app = FastAPI(
                title="LangGraph OpenAI Compatible API",
                description="An OpenAI-compatible API for LangGraph",
                version=get_version(),
            )
        self.app = app

        if graphs is None:
            logger.info("Graphs not provided, using default simple graph")
            default_graph_config = GraphConfig(
                graph=simple_graph,
                streamable_node_names=["generate"],
            )
            self.graph_registry = GraphRegistry(
                registry={"simple-graph": default_graph_config}
            )
        elif isinstance(graphs, GraphRegistry):
            logger.info("Using provided GraphRegistry instance")
            self.graph_registry = graphs
        else:
            raise TypeError(
                "Invalid type for graphs parameter. Expected GraphRegistry or None."
            )

        # Attach the registry to the host app for callers that inspect app state.
        self.app.state.graph_registry = self.graph_registry

        # Configure CORS if requested
        if configure_cors:
            self._configure_cors()

        logger.info(
            f"Initialized LangchainOpenaiApiServe with {len(self.graph_registry.registry)} graphs"
        )
        logger.info(
            f"Available graphs: {', '.join(self.graph_registry.get_graph_names())}"
        )

    def bind_openai_chat_completion(self, prefix: str | None = None):
        """Mount OpenAI-compatible endpoints on the host FastAPI app.

        Args:
            prefix: Optional; The URL prefix for the OpenAI-compatible endpoints.
                Defaults to settings.OPENAI_API_PREFIX.
        """
        if prefix is None:
            prefix = settings.OPENAI_API_PREFIX
        else:
            prefix = normalize_openai_api_prefix(prefix)

        docs_url = "/docs" if settings.OPENAI_API_DOCS_ENABLED else None
        redoc_url = "/redoc" if settings.OPENAI_API_DOCS_ENABLED else None
        openapi_url = "/openapi.json" if settings.OPENAI_API_DOCS_ENABLED else None

        openai_app = FastAPI(
            title="LangGraph OpenAI Compatible API",
            description="An OpenAI-compatible API for LangGraph",
            version=get_version(),
            docs_url=docs_url,
            redoc_url=redoc_url,
            openapi_url=openapi_url,
        )
        # Dependencies in mounted routes resolve against the mounted app.
        openai_app.state.graph_registry = self.graph_registry
        configure_openai_error_handlers(openai_app)
        openai_app.include_router(chat_views.router)
        openai_app.include_router(health_views.router)
        openai_app.include_router(models_views.router)

        self.app.mount(prefix, openai_app, name="openai")

        logger.info(f"Bound OpenAI chat completion endpoints with prefix: {prefix}")

        return self
