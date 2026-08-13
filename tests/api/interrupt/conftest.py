from typing import Any

import pytest
from anyio import Event
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from tests.graph.support.interrupt import (
    InterruptAnswerState,
    make_interrupt_graph,
    make_parallel_interrupt_graph,
    make_parallel_nested_interrupt_graph,
    make_sequential_interrupt_graph,
    make_sequential_nested_interrupt_graph,
)

from .support import (
    CHECKPOINT_SCOPE_HEADER,
    CONCURRENT_MODEL,
    INVALID_PAYLOAD_MODEL,
    MODEL,
    NESTED_MODEL,
    NESTED_SEQUENTIAL_MODEL,
    PARALLEL_MODEL,
    SEQUENTIAL_MODEL,
)


@pytest.fixture
def fastapi_app(sqlite_checkpointer: AsyncSqliteSaver) -> FastAPI:
    coordinator = InMemoryRunCoordinator()
    resume_entered = Event()
    resume_release = Event()
    side_effects = {"count": 0}

    async def concurrent_approval(
        _state: InterruptAnswerState,
    ) -> dict[str, list[str]]:
        decision = interrupt({"question": "concurrent"})
        side_effects["count"] += 1
        resume_entered.set()
        await resume_release.wait()
        return {"answers": [str(decision)]}

    concurrent_graph = (
        StateGraph(InterruptAnswerState)
        .add_node("approve", concurrent_approval)
        .add_edge(START, "approve")
        .add_edge("approve", END)
        .compile(checkpointer=sqlite_checkpointer)
    )

    def empty_answers(_request: Any, _messages: Any) -> dict[str, list[str]]:
        return {"answers": []}

    def render_answers(output: dict[str, list[str]]) -> str:
        return ",".join(output["answers"])

    def render_sorted_answers(output: dict[str, list[str]]) -> str:
        return ",".join(sorted(output["answers"]))

    def interrupt_config(graph: Any, **kwargs: Any) -> GraphConfig:
        return GraphConfig(
            graph=graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            run_coordinator=coordinator,
            **kwargs,
        )

    graph_registry = GraphRegistry(
        registry={
            MODEL: interrupt_config(
                make_interrupt_graph(checkpointer=sqlite_checkpointer),
            ),
            PARALLEL_MODEL: interrupt_config(
                make_parallel_interrupt_graph(sqlite_checkpointer),
                request_to_input=empty_answers,
                output_to_text=render_sorted_answers,
            ),
            SEQUENTIAL_MODEL: interrupt_config(
                make_sequential_interrupt_graph(sqlite_checkpointer),
                request_to_input=empty_answers,
                output_to_text=render_answers,
            ),
            CONCURRENT_MODEL: interrupt_config(
                concurrent_graph,
                request_to_input=empty_answers,
                output_to_text=lambda output: output["answers"][0],
            ),
            INVALID_PAYLOAD_MODEL: interrupt_config(
                make_interrupt_graph(
                    {"value": float("nan")},
                    checkpointer=sqlite_checkpointer,
                ),
            ),
            NESTED_MODEL: interrupt_config(
                make_parallel_nested_interrupt_graph(sqlite_checkpointer),
                request_to_input=empty_answers,
                output_to_text=render_sorted_answers,
            ),
            NESTED_SEQUENTIAL_MODEL: interrupt_config(
                make_sequential_nested_interrupt_graph(sqlite_checkpointer),
                request_to_input=empty_answers,
                output_to_text=render_answers,
            ),
        }
    )
    app = (
        LanggraphOpenaiServe(
            graphs=graph_registry,
            checkpoint_scope=lambda request: request.headers.get(
                CHECKPOINT_SCOPE_HEADER,
                "default",
            ),
        )
        .bind_openai_api()
        .app
    )
    app.state.resume_entered = resume_entered
    app.state.resume_release = resume_release
    app.state.side_effects = side_effects
    return app
