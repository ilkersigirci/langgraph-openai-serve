"""Exercise plot persistence across PostgreSQL runtime restarts."""

import os
import uuid

import pytest
from langchain_core.messages import HumanMessage

from lgos_demo_api.checkpointer import postgres_runtime, setup_postgres_schema
from lgos_demo_api.graphs.persistent_plot import (
    ARTIFACT_KEY,
    PersistentPlotContext,
    PersistentPlotSettings,
    _thread_namespace,
    create_persistent_plot_graph,
)

POSTGRES_URI = os.environ.get("DEMO_API_TEST_POSTGRES_URI")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        POSTGRES_URI is None,
        reason="DEMO_API_TEST_POSTGRES_URI is required",
    ),
]


async def test_persistent_plot_survives_runtime_restart() -> None:
    assert POSTGRES_URI is not None
    await setup_postgres_schema(POSTGRES_URI)
    context = PersistentPlotContext(
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        settings=PersistentPlotSettings(),
    )

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            graph = create_persistent_plot_graph(runtime.store)
            await graph.ainvoke(
                {"messages": [HumanMessage(content="Set Q3 to 250.")]},
                context=context,
            )

        async with postgres_runtime(POSTGRES_URI) as runtime:
            graph = create_persistent_plot_graph(runtime.store)
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Which quarter is highest?")]},
                context=context,
            )
            assert result["messages"][-1].content == "Q3 is highest at $250k."
    finally:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            await runtime.store.adelete(_thread_namespace(context), ARTIFACT_KEY)
