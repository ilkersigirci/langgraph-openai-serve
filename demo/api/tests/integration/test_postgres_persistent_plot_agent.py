"""Exercise persistent plot agent data across PostgreSQL runtime restarts."""

import os
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from lgos_demo_api.checkpointer import postgres_runtime, setup_postgres_schema
from lgos_demo_api.graphs.persistent_plot_agent import (
    ARTIFACT_KEY,
    PersistentPlotAgentContext,
    PersistentPlotAgentSettings,
    PlotDocument,
    _thread_namespace,
    create_persistent_plot_agent,
)

POSTGRES_URI = os.environ.get("DEMO_API_TEST_POSTGRES_URI")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        POSTGRES_URI is None,
        reason="DEMO_API_TEST_POSTGRES_URI is required",
    ),
]


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


async def test_persistent_plot_agent_survives_runtime_restart(
    make_tool_calling_model: Callable[..., BaseChatModel],
) -> None:
    assert POSTGRES_URI is not None
    await setup_postgres_schema(POSTGRES_URI)
    context = PersistentPlotAgentContext(
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        settings=PersistentPlotAgentSettings(),
    )

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            graph = create_persistent_plot_agent(
                runtime.store,
                make_tool_calling_model(
                    _tool_call(
                        "update_quarterly_revenue",
                        {"updates": [{"quarter": "Q3", "revenue": 250}]},
                        "update-1",
                    ),
                    AIMessage(content="Updated Q3."),
                ),
            )
            await graph.ainvoke(
                {"messages": [HumanMessage(content="Set Q3 to 250.")]},
                context=context,
            )

        async with postgres_runtime(POSTGRES_URI) as runtime:
            item = await runtime.store.aget(
                _thread_namespace(context),
                ARTIFACT_KEY,
            )
            assert item is not None
            assert PlotDocument.model_validate(item.value).q3 == 250
    finally:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            await runtime.store.adelete(_thread_namespace(context), ARTIFACT_KEY)
