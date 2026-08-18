import operator

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import CustomStreamPart

from langgraph_openai_serve.core.settings import Settings
from langgraph_openai_serve.graph import utils as graph_utils
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphNotFoundError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.runner import (
    invoke_run,
    run_langgraph,
)
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    prepare_run,
)
from tests.graph.support.interrupt import make_interrupt_graph
from tests.graph.support.message import make_message_graph


class RecordingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.starts = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.starts += 1


@pytest.fixture
def mock_langfuse_callback(monkeypatch: pytest.MonkeyPatch) -> RecordingCallback:
    callback = RecordingCallback()
    monkeypatch.setattr(
        graph_utils, "settings", Settings.model_construct(ENABLE_LANGFUSE=True)
    )
    monkeypatch.setattr(
        graph_utils,
        "get_langfuse_callback",
        lambda: callback,
    )
    return callback


@pytest.mark.parametrize(
    "has_explicit_callbacks",
    [True, False],
)
async def test_enabled_langfuse_is_added_to_graph_run(
    make_request,
    mock_langfuse_callback: RecordingCallback,
    has_explicit_callbacks: bool,
) -> None:
    recording_callback = RecordingCallback() if has_explicit_callbacks else None
    runtime_callbacks = [recording_callback] if recording_callback else None

    graph_config = GraphConfig(
        graph=make_message_graph("hello"),
        description="DUMMY",
        runtime_callbacks=runtime_callbacks,
    )
    graph_registry = GraphRegistry(
        registry={
            "messages": graph_config,
        },
    )
    chat_request = make_request("messages")

    invocation = await run_langgraph(
        "messages",
        chat_request.messages,
        graph_registry,
        chat_request,
    )

    assert invocation.output == "hello"
    assert mock_langfuse_callback.starts == 1

    if recording_callback:
        assert recording_callback.starts == 1
        assert graph_config.runtime_callbacks == [recording_callback]
    else:
        assert graph_config.runtime_callbacks is None


async def test_runtime_callbacks_reach_interrupt_runnable_config_without_mutation(
    make_request,
    sqlite_checkpointer,
) -> None:
    recording_callback = RecordingCallback()
    runtime_callbacks = [recording_callback]
    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
        runtime_callbacks=runtime_callbacks,
        run_coordinator=InMemoryRunCoordinator(),
    )
    graph_registry = GraphRegistry(registry={"interruptible": graph_config})
    request = make_request("interruptible")

    run = await prepare_run(
        "interruptible",
        request.messages,
        graph_registry,
        request,
    )
    try:
        assert run.runnable_config is not None
        assert run.runnable_config["callbacks"] == [recording_callback]
        assert graph_config.runtime_callbacks == [recording_callback]
        assert runtime_callbacks == [recording_callback]
    finally:
        await run.aclose()


async def test_unknown_model_raises_graph_not_found_error(make_request) -> None:
    chat_request = make_request("missing")
    graph_registry = GraphRegistry(
        registry={
            "known": GraphConfig(
                graph=make_message_graph("hello"),
                description="DUMMY",
            ),
        }
    )

    with pytest.raises(GraphNotFoundError, match="Graph 'missing' not found"):
        await run_langgraph(
            "missing",
            chat_request.messages,
            graph_registry,
            chat_request,
        )


async def test_invoke_run_collects_generic_custom_events() -> None:
    payload = {"type": "status", "data": {"message": "Searching"}}

    async def graph_events():
        yield {"type": "values", "ns": (), "data": {"answer": ""}}
        yield {"type": "custom", "ns": (), "data": payload}
        yield {"type": "values", "ns": (), "data": {"answer": "done"}}

    class Graph:
        output_channels = ("answer",)

        def astream(self, *args, **kwargs):
            return graph_events()

    graph = Graph()
    run = GraphRun(
        config=GraphConfig(
            graph=lambda: graph,
            description="DUMMY",
            output_to_text=operator.itemgetter("answer"),
        ),
        graph=graph,
        inputs={},
        context=None,
        runnable_config=None,
        run_id=None,
    )

    invocation = await invoke_run(run)

    assert invocation.output == "done"
    assert invocation.custom_events == (
        CustomStreamPart(type="custom", ns=(), data=payload),
    )
