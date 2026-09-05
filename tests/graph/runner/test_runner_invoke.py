import pytest
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph

from langgraph_openai_serve.core.logging import (
    begin_log_context,
    get_log_context,
    reset_log_context,
)
from langgraph_openai_serve.core.settings import Settings
from langgraph_openai_serve.graph import utils as graph_utils
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
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
        self.root_metadata: list[dict[str, object]] = []

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.starts += 1

    def on_chain_start(
        self,
        *args,
        parent_run_id=None,
        metadata=None,
        **kwargs,
    ) -> None:
        if parent_run_id is None:
            self.root_metadata.append(dict(metadata or {}))


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
    request = make_request("messages")

    invocation = await run_langgraph(
        request, [HumanMessage(content="question")], graph_registry
    )

    assert invocation.output.text == "hello"
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
    request = make_request(
        "interruptible",
        metadata={"session_id": "conversation-123"},
    )

    run = await prepare_run(
        request,
        [HumanMessage(content="question")],
        graph_registry,
    )
    try:
        assert run.runnable_config is not None
        assert run.runnable_config["callbacks"][0] is recording_callback
        assert isinstance(
            run.runnable_config["callbacks"][1],
            UsageMetadataCallbackHandler,
        )
        assert run.runnable_config["run_name"] == "lgos.graph_run"
        assert run.runnable_config["metadata"]["lgos.model"] == "interruptible"
        assert run.runnable_config["metadata"]["lgos.operation_id"] is not None
        assert (
            run.runnable_config["metadata"]["langfuse_session_id"] == "conversation-123"
        )
        assert "run_id" not in run.runnable_config
        assert graph_config.runtime_callbacks == [recording_callback]
        assert runtime_callbacks == [recording_callback]
    finally:
        await run.aclose()


async def test_interrupt_callback_observes_native_checkpoint_metadata(
    make_request,
    sqlite_checkpointer,
) -> None:
    recording_callback = RecordingCallback()
    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
        runtime_callbacks=[recording_callback],
        run_coordinator=InMemoryRunCoordinator(),
    )
    graph_registry = GraphRegistry(registry={"interruptible": graph_config})
    request = make_request(
        "interruptible",
        metadata={"session_id": "conversation-123"},
    )

    await run_langgraph(request, [HumanMessage(content="question")], graph_registry)

    assert recording_callback.root_metadata
    metadata = recording_callback.root_metadata[0]
    assert metadata["lgos.model"] == "interruptible"
    assert metadata["lgos.operation_id"]
    assert metadata["langfuse_session_id"] == "conversation-123"
    assert metadata["thread_id"]
    assert get_log_context() == {}


async def test_runnable_config_contains_request_correlation_metadata(
    make_request,
) -> None:
    recording_callback = RecordingCallback()
    graph_config = GraphConfig(
        graph=make_message_graph("hello"),
        description="DUMMY",
        runtime_callbacks=[recording_callback],
    )
    graph_registry = GraphRegistry(registry={"messages": graph_config})
    request = make_request(
        "messages",
        metadata={
            "session_id": "conversation-123",
            "unrelated": "not callback metadata",
        },
    )
    token = begin_log_context("request-123")

    try:
        run = await prepare_run(
            request,
            [HumanMessage(content="question")],
            graph_registry,
        )
    finally:
        reset_log_context(token)

    try:
        assert run.runnable_config is not None
        assert run.runnable_config["run_name"] == "lgos.graph_run"
        assert run.runnable_config["metadata"] == {
            "lgos.model": "messages",
            "lgos.request_id": "request-123",
            "langfuse_session_id": "conversation-123",
        }
        assert "run_id" not in run.runnable_config
    finally:
        await run.aclose()


async def test_operation_id_is_bound_before_interrupt_preparation_fails(
    make_request,
    sqlite_checkpointer,
) -> None:
    error_message = "context failed"

    def fail_context(_request, _settings):
        raise RuntimeError(error_message)

    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
        context_factory=fail_context,
        run_coordinator=InMemoryRunCoordinator(),
    )
    graph_registry = GraphRegistry(registry={"interruptible": graph_config})
    request = make_request("interruptible")
    token = begin_log_context("request-123")

    try:
        with pytest.raises(RuntimeError, match=error_message):
            await prepare_run(
                request,
                [HumanMessage(content="question")],
                graph_registry,
            )

        assert get_log_context()["operation_id"]
    finally:
        reset_log_context(token)


async def test_standard_graph_rejects_interrupt_run_coordinator() -> None:
    graph_config = GraphConfig(
        graph=make_message_graph("hello"),
        description="DUMMY",
        run_coordinator=InMemoryRunCoordinator(),
    )

    with pytest.raises(GraphConfigurationError, match="interrupt-enabled"):
        await graph_config.resolve_graph()


async def test_unknown_model_raises_graph_not_found_error(make_request) -> None:
    request = make_request("missing")
    graph_registry = GraphRegistry(
        registry={
            "known": GraphConfig(
                graph=make_message_graph("hello"),
                description="DUMMY",
            ),
        }
    )

    with pytest.raises(GraphNotFoundError, match="Graph 'missing' not found"):
        await run_langgraph(request, [HumanMessage(content="question")], graph_registry)


async def test_invoke_run_ignores_generic_custom_events() -> None:
    payload = {"type": "status", "data": {"message": "Searching"}}

    async def answer(_state):
        get_stream_writer()(payload)
        return {"messages": [AIMessage(content="done")]}

    graph = (
        StateGraph(dict)
        .add_node("answer", answer)
        .set_entry_point("answer")
        .set_finish_point("answer")
        .compile()
    )
    run = GraphRun(
        config=GraphConfig(
            graph=graph,
            description="DUMMY",
        ),
        graph=graph,
        inputs={},
        context=None,
        runnable_config=None,
        run_id=None,
    )

    invocation = await invoke_run(run)

    assert invocation.output.text == "done"
