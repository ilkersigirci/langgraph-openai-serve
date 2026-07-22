"""LangGraph OpenAI API Serve.

This module provides a server class that connects LangGraph instances to an
OpenAI-compatible API. It allows users to register their LangGraph instances
and expose them through a mounted FastAPI sub-application.

Examples:
    >>> from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
    >>> from fastapi import FastAPI
    >>> from your_graphs import simple_graph_1, simple_graph_2
    >>>
    >>> app = FastAPI(title="LangGraph OpenAI API")
    >>> graphs = GraphRegistry(
    ...     registry={
    ...         "simple_graph_1": GraphConfig(graph=simple_graph_1),
    ...         "simple_graph_2": GraphConfig(graph=simple_graph_2),
    ...     }
    ... )
    >>> graph_serve = LanggraphOpenaiServe(
    ...     app=app,
    ...     graphs=graphs,
    ... )
    >>> graph_serve.bind_openai_api()
"""

import logging

from fastapi import FastAPI

from langgraph_openai_serve.api.chat import views as chat_views
from langgraph_openai_serve.api.health import views as health_views
from langgraph_openai_serve.api.models import views as models_views
from langgraph_openai_serve.core.errors import configure_openai_error_handlers
from langgraph_openai_serve.core.settings import normalize_openai_api_prefix, settings
from langgraph_openai_serve.core.version import get_version
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = logging.getLogger(__name__)


class LanggraphOpenaiServe:
    """Server class to connect LangGraph instances with an OpenAI-compatible API.

    This class serves as a bridge between LangGraph instances and an OpenAI-compatible API.
    It allows users to register their LangGraph instances and expose them
    through an OpenAI-compatible sub-application mounted on a FastAPI host app.

    Attributes:
        app: The host FastAPI application to mount the OpenAI API on.
        graph_registry: The populated GraphRegistry containing the graphs to serve.
    """

    def __init__(
        self,
        graphs: GraphRegistry,
        app: FastAPI | None = None,
    ):
        """Initialize the server with a FastAPI app and a populated graph registry.

        Args:
            app: The host FastAPI application to mount the OpenAI API on. If None,
                a new FastAPI app will be created.
            graphs: A GraphRegistry instance containing the graphs to serve.

        Raises:
            TypeError: If graphs is not a GraphRegistry instance.
        """
        if not isinstance(graphs, GraphRegistry):
            raise TypeError(
                "Invalid type for graphs parameter. Expected GraphRegistry."
            )

        if app is None:
            app = FastAPI(
                title="LangGraph OpenAI Compatible API",
                description="An OpenAI-compatible API for LangGraph",
                version=get_version(),
            )
        self.app: FastAPI = app

        logger.info("Using provided GraphRegistry instance")
        self.graph_registry = graphs

        # Host integrations can inspect registered graphs without traversing the
        # mounted OpenAI sub-application.
        self.app.state.graph_registry = self.graph_registry

        logger.info(
            f"Initialized LanggraphOpenaiServe with {len(self.graph_registry.registry)} graphs"
        )
        logger.info(
            f"Available graphs: {', '.join(self.graph_registry.get_graph_names())}"
        )

    def bind_openai_api(self, prefix: str | None = None):
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
