"""
LangGraph OpenAI API Serve.

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
    ...         "simple_graph_1": GraphConfig(
    ...             graph=simple_graph_1,
    ...             description="First simple graph.",
    ...         ),
    ...         "simple_graph_2": GraphConfig(
    ...             graph=simple_graph_2,
    ...             description="Second simple graph.",
    ...         ),
    ...     }
    ... )
    >>> graph_serve = LanggraphOpenaiServe(
    ...     app=app,
    ...     graphs=graphs,
    ... )
    >>> graph_serve.bind_openai_api()

"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from starlette.middleware import Middleware
from starlette.routing import Mount

from langgraph_openai_serve.api.chat import views as chat_views
from langgraph_openai_serve.api.health import views as health_views
from langgraph_openai_serve.api.middleware import RequestContextMiddleware
from langgraph_openai_serve.api.models import views as models_views
from langgraph_openai_serve.core.errors import configure_openai_error_handlers
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.core.settings import normalize_openai_api_prefix, settings
from langgraph_openai_serve.core.version import get_version
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = get_logger(__name__)


class LanggraphOpenaiServe:
    """
    Server class to connect LangGraph instances with an OpenAI-compatible API.

    This class serves as a bridge between LangGraph instances and an OpenAI-compatible API.
    It allows users to register their LangGraph instances and expose them
    through an OpenAI-compatible sub-application mounted on a FastAPI host app.

    Attributes:
        app: The host FastAPI application to mount the OpenAI API on.
        graph_registry: The populated GraphRegistry containing the graphs to serve.
        openai_app: The mounted OpenAI-compatible FastAPI application.

    """

    def __init__(
        self,
        graphs: GraphRegistry,
        app: FastAPI | None = None,
        checkpoint_scope: Callable[[Request], str | Awaitable[str]] | None = None,
    ) -> None:
        """
        Initialize the server with a FastAPI app and a populated graph registry.

        Args:
            app: The host FastAPI application to mount the OpenAI API on. If None,
                a new FastAPI app will be created.
            graphs: A GraphRegistry instance containing the graphs to serve.
            checkpoint_scope: Optional server-trusted resolver used to isolate
                interrupt checkpoints by deployment or authenticated principal.

        Raises:
            TypeError: If graphs is not a GraphRegistry instance.

        """
        if not isinstance(graphs, GraphRegistry):
            msg = "Invalid type for graphs parameter. Expected GraphRegistry."
            raise TypeError(msg)

        if app is None:
            app = FastAPI(
                title="LangGraph OpenAI Compatible API",
                description="An OpenAI-compatible API for LangGraph",
                version=get_version(),
            )
        self.app: FastAPI = app
        self._openai_app: FastAPI | None = None
        self.checkpoint_scope = checkpoint_scope or (lambda _request: "default")

        self.graph_registry = graphs

        # Host integrations can inspect registered graphs without traversing the
        # mounted OpenAI sub-application.
        self.app.state.graph_registry = self.graph_registry
        self.app.state.checkpoint_scope = self.checkpoint_scope

        logger.info(
            "server.initialized",
            extra={"graph_count": len(self.graph_registry.registry)},
        )

    @property
    def openai_app(self) -> FastAPI:
        """The mounted OpenAI-compatible FastAPI application."""
        if self._openai_app is None:
            msg = "OpenAI API is not bound. Call bind_openai_api() first."
            raise RuntimeError(msg)
        return self._openai_app

    def bind_openai_api(self, prefix: str | None = None) -> "LanggraphOpenaiServe":
        """
        Mount OpenAI-compatible endpoints on the host FastAPI app.

        Args:
            prefix: Optional; The URL prefix for the OpenAI-compatible endpoints.
                Defaults to settings.OPENAI_API_PREFIX.

        """
        prefix = (
            normalize_openai_api_prefix(prefix)
            if prefix is not None
            else settings.OPENAI_API_PREFIX
        )

        openai_app = FastAPI(
            title="LangGraph OpenAI Compatible API",
            description="An OpenAI-compatible API for LangGraph",
            version=get_version(),
            **settings.fastapi_docs_kwargs,
        )
        # Dependencies in mounted routes resolve against the mounted app.
        openai_app.state.graph_registry = self.graph_registry
        openai_app.state.checkpoint_scope = self.checkpoint_scope
        configure_openai_error_handlers(openai_app)
        openai_app.include_router(chat_views.router)
        openai_app.include_router(health_views.router)
        openai_app.include_router(models_views.router)

        self.app.router.routes.append(
            Mount(
                prefix,
                app=openai_app,
                name="openai",
                middleware=[Middleware(RequestContextMiddleware)],
            )
        )
        self._openai_app = openai_app

        logger.info("server.api_bound", extra={"prefix": prefix})

        return self
