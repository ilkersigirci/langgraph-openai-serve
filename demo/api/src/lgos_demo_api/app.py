"""
FastAPI application for LangGraph with OpenAI compatible API.

This module provides a demo FastAPI application that exposes example LangGraph
graphs through the OpenAI-compatible API.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe
from openai import AsyncOpenAI

from lgos_demo_api.background import create_background_components
from lgos_demo_api.checkpointer import postgres_runtime
from lgos_demo_api.graphs.advanced_graph import (
    OpenAICompatibleKnowledgeBase,
    create_advanced_graph,
    create_advanced_graph_config,
    create_model,
)
from lgos_demo_api.graphs.background_report import (
    create_background_report_config,
    create_background_report_graph,
)
from lgos_demo_api.graphs.citations import citation_graph_config
from lgos_demo_api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from lgos_demo_api.graphs.custom_events import custom_event_showcase_graph_config
from lgos_demo_api.graphs.custom_io import custom_io_graph_config
from lgos_demo_api.graphs.file_input import file_input_graph_config
from lgos_demo_api.graphs.interruptible import (
    create_interruptible_graph,
    create_interruptible_graph_config,
)
from lgos_demo_api.graphs.lgos_rag import lgos_rag_graph_config
from lgos_demo_api.graphs.mcp_mock import mcp_mock_graph_config
from lgos_demo_api.graphs.mcp_postgres import mcp_postgres_graph_config
from lgos_demo_api.graphs.multi_node_streaming import (
    multi_node_streaming_graph_config,
)
from lgos_demo_api.graphs.persistent_plot_agent import (
    create_persistent_plot_agent,
    create_persistent_plot_agent_config,
)
from lgos_demo_api.graphs.response_outcomes import response_outcome_graph_config
from lgos_demo_api.graphs.server_tool import server_tool_graph_config
from lgos_demo_api.graphs.simple import simple_graph_config
from lgos_demo_api.graphs.simple_external_tools import (
    simple_external_tools_graph_config,
)
from lgos_demo_api.graphs.status_events import status_event_graph_config
from lgos_demo_api.logging import LOGGING_CONFIG
from lgos_demo_api.otel import instrument_fastapi_app
from lgos_demo_api.settings import settings

logger = logging.getLogger(__name__)


def _vector_store_connection() -> tuple[str, str]:
    """Resolve storage credentials without leaking the model provider's key."""
    if settings.VECTOR_STORE_BASE_URL:
        return (
            settings.VECTOR_STORE_BASE_URL,
            settings.VECTOR_STORE_API_KEY or "DUMMY",
        )
    return (
        settings.OPENAI_BASE_URL,
        settings.VECTOR_STORE_API_KEY or settings.OPENAI_API_KEY,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    This function handles the startup and shutdown events for the application.

    Args:
        app: The FastAPI application.

    """
    logger.info("demo.server.starting")
    vector_store_base_url, vector_store_api_key = _vector_store_connection()

    async with (
        postgres_runtime(settings.POSTGRES_URI) as runtime,
        httpx2.AsyncClient(timeout=60) as upstream_http,
        AsyncOpenAI(
            base_url=vector_store_base_url,
            api_key=vector_store_api_key,
            default_headers=(
                {"x-bf-api-key": settings.VECTOR_STORE_BIFROST_KEY_NAME}
                if settings.VECTOR_STORE_BIFROST_KEY_NAME
                else None
            ),
            http_client=upstream_http,
            max_retries=0,
        ) as vector_store_client,
        AsyncOpenAI(
            base_url=settings.FILES_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
        ) as files_client,
    ):
        app.state.interruptible_graph = create_interruptible_graph(runtime.checkpointer)
        app.state.background_report_graph = create_background_report_graph(
            runtime.checkpointer
        )
        app.state.run_coordinator = runtime.run_coordinator
        app.state.persistent_plot_agent = create_persistent_plot_agent(runtime.store)
        knowledge = (
            OpenAICompatibleKnowledgeBase(
                vector_store_client,
                settings.VECTOR_STORE_ID,
            )
            if settings.VECTOR_STORE_ID
            else None
        )
        app.state.advanced_graph = create_advanced_graph(
            model=create_model(upstream_http),
            knowledge=knowledge,
            files=files_client,
            checkpointer=runtime.checkpointer,
            store=runtime.store,
        )
        background = None
        if settings.BACKGROUND_ENABLED:
            components = create_background_components(
                app.state.graph_registry,
                runtime.response_store,
            )
            background = components.backend
            app.state.background_components = components
        app.state.background_backend = background
        app.state.lgos_openai_app.state.background_backend = background

        yield

        app.state.background_backend = None
        app.state.lgos_openai_app.state.background_backend = None

    logger.info("demo.server.stopped")


def create_custom_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        A configured FastAPI application.

    """
    app = FastAPI(
        title="Demo",
        version="0.0.1",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        # Local browser demos may use arbitrary origins; deployments must replace
        # this wildcard with their trusted origins. The demo has no cookie-based
        # authentication, so wildcard origins do not need credentials enabled.
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    graph_registry = GraphRegistry(
        registry={
            "advanced-graph": create_advanced_graph_config(
                lambda: app.state.advanced_graph,
                lambda key: app.state.run_coordinator(key),
            ),
            "background-report-agent": create_background_report_config(
                lambda: app.state.background_report_graph,
                lambda key: app.state.run_coordinator(key),
            ),
            "citation-events": citation_graph_config,
            "file-input": file_input_graph_config,
            "simple-graph": simple_graph_config,
            "server-tool": server_tool_graph_config,
            "lgos-rag": lgos_rag_graph_config,
            "custom-input-output-context": custom_io_graph_config,
            "mcp-mock": mcp_mock_graph_config,
            "mcp-postgres": mcp_postgres_graph_config,
            "complex-subgraphs": create_complex_subgraphs_graph_config(),
            "multi-node-streaming": multi_node_streaming_graph_config,
            "custom-event-showcase": custom_event_showcase_graph_config,
            "status-events": status_event_graph_config,
            "response-outcomes": response_outcome_graph_config,
            "persistent-plot-agent": create_persistent_plot_agent_config(
                lambda: app.state.persistent_plot_agent,
            ),
            "simple-graph-external-tools": simple_external_tools_graph_config,
            "interruptible-approval": create_interruptible_graph_config(
                # We use lambdas here because app.state is populated asynchronously
                # during the FastAPI lifespan event. Eagerly evaluating app.state
                # attributes at registry initialization time would raise an
                # AttributeError since the lifespan has not executed yet.
                lambda: app.state.interruptible_graph,
                lambda key: app.state.run_coordinator(key),
            ),
        }
    )

    graph_serve = LanggraphOpenaiServe(
        app=app,
        graphs=graph_registry,
    )

    graph_serve.bind_openai_api()
    app.state.lgos_openai_app = graph_serve.openai_app
    instrument_fastapi_app(graph_serve.openai_app)

    return app


def main() -> None:
    """Run the demo API with JSON logging."""
    import uvicorn

    uvicorn.run(
        "lgos_demo_api.app:create_custom_app",
        factory=True,
        host="0.0.0.0",
        port=settings.PORT,
        access_log=False,
        log_config=LOGGING_CONFIG,
    )


if __name__ == "__main__":
    main()
