"""The FastAPI application that ``lgos serve`` runs."""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langgraph_openai_serve.background import (
    BackgroundBackend,
    InMemoryBackgroundBackend,
)
from langgraph_openai_serve.openai_server import LanggraphOpenaiServe
from langgraph_openai_serve.server.hatchet import create_hatchet_backend
from langgraph_openai_serve.server.runtime import (
    RegistryFactory,
    expire_paused_runs,
    open_registry,
    open_resources,
)
from langgraph_openai_serve.server.settings import ServerSettings


def create_app(
    registry: RegistryFactory,
    *,
    settings: ServerSettings | None = None,
) -> FastAPI:
    """Serve the factory's registry under the OpenAI prefix once the app starts."""
    settings = settings or ServerSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        async with AsyncExitStack() as stack:
            resources = await stack.enter_async_context(open_resources(settings))
            graphs = await stack.enter_async_context(open_registry(registry, resources))
            background: BackgroundBackend | None = None
            if settings.BACKGROUND == "memory":
                memory = InMemoryBackgroundBackend(graphs)
                await stack.enter_async_context(memory.lifespan(app))
                background = memory
            elif settings.BACKGROUND == "hatchet":
                background = create_hatchet_backend()
            tasks = await stack.enter_async_context(anyio.create_task_group())
            # Registered after the task group, this runs before the group exits.
            stack.callback(tasks.cancel_scope.cancel)
            if settings.INTERRUPT_SWEEP_INTERVAL_MINUTES:
                tasks.start_soon(expire_paused_runs, resources, settings)
            # The registry exists only after its resources open, so the OpenAI
            # routes are mounted at startup and removed at shutdown; otherwise a
            # restarted app would keep serving the closed registry first.
            LanggraphOpenaiServe(
                registry=graphs, app=app, background=background
            ).bind_openai_api()
            stack.callback(app.router.routes.remove, app.router.routes[-1])
            yield

    app = FastAPI(
        title="LangGraph OpenAI Compatible API",
        lifespan=lifespan,
        openapi_url=None,
        # FastAPI records requests through the global providers; export belongs
        # to `opentelemetry-instrument`, so FastAPI must not add its own.
        telemetry={
            "auto_configure": False,
            "exclude": lambda scope: scope["path"].endswith("/health"),
        },
    )
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    return app
