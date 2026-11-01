"""Mount a graph registry's OpenAI-compatible API on a FastAPI application."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from starlette.middleware import Middleware
from starlette.routing import Mount

from langgraph_openai_serve.api.chat import views as chat_views
from langgraph_openai_serve.api.health import views as health_views
from langgraph_openai_serve.api.middleware import RequestContextMiddleware
from langgraph_openai_serve.api.models import views as models_views
from langgraph_openai_serve.api.responses import views as responses_views
from langgraph_openai_serve.background import BackgroundBackend
from langgraph_openai_serve.core.errors import configure_openai_error_handlers
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.core.settings import normalize_openai_api_prefix, settings
from langgraph_openai_serve.core.version import VERSION
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = get_logger(__name__)


class LanggraphOpenaiServe:
    """Mount a graph registry's OpenAI-compatible API on a FastAPI host app."""

    def __init__(
        self,
        registry: GraphRegistry,
        app: FastAPI | None = None,
        checkpoint_scope: Callable[[Request], str | Awaitable[str]] | None = None,
        background: BackgroundBackend | None = None,
    ) -> None:
        """
        Initialize the server with a FastAPI app and a populated graph registry.

        Args:
            registry: The graphs to serve and their interrupt run coordinator.
            app: The host FastAPI application to mount the OpenAI API on. If None,
                a new FastAPI app will be created.
            checkpoint_scope: Optional server-trusted resolver used to isolate
                interrupt checkpoints by deployment or authenticated principal.
            background: Optional polling-only background backend.

        """
        if app is None:
            app = FastAPI(
                title="LangGraph OpenAI Compatible API",
                description="An OpenAI-compatible API for LangGraph",
                version=VERSION,
            )
        self.app: FastAPI = app
        self._openai_app: FastAPI | None = None
        self.checkpoint_scope = checkpoint_scope or (lambda _request: "default")
        self.background = background
        self.graph_registry = registry

        # Host integrations can inspect registered graphs without traversing the
        # mounted OpenAI sub-application.
        self.app.state.graph_registry = registry

        logger.info("server.initialized", extra={"graph_count": len(registry.graphs)})

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
            prefix: URL prefix for the OpenAI routes; defaults to the
                ``LGOS_OPENAI_API_PREFIX`` setting.

        """
        prefix = (
            normalize_openai_api_prefix(prefix)
            if prefix is not None
            else settings.OPENAI_API_PREFIX
        )

        openai_app = FastAPI(
            title="LangGraph OpenAI Compatible API",
            description="An OpenAI-compatible API for LangGraph",
            version=VERSION,
            # FastAPI's native telemetry belongs to the host app, so the host's
            # settings, including turning it off, also cover the mounted routes.
            telemetry={"tracing": False, "metrics": False, "logs": False},
            # Without an OpenAPI URL, FastAPI serves no /docs or /redoc either.
            openapi_url="/openapi.json" if settings.OPENAI_API_DOCS_ENABLED else None,
        )
        # Dependencies in mounted routes resolve against the mounted app.
        openai_app.state.graph_registry = self.graph_registry
        openai_app.state.checkpoint_scope = self.checkpoint_scope
        openai_app.state.background_backend = self.background
        configure_openai_error_handlers(openai_app)
        openai_app.include_router(chat_views.router)
        openai_app.include_router(health_views.router)
        openai_app.include_router(models_views.router)
        openai_app.include_router(responses_views.router)

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
