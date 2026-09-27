import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from langgraph_openai_serve.core.errors import InvalidRequestError
from langgraph_openai_serve.core.logging import (
    begin_log_context,
    get_log_context,
    reset_log_context,
)
from langgraph_openai_serve.core.settings import Settings
from langgraph_openai_serve.graph import run as graph_run
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.run import prepare_run
from langgraph_openai_serve.graph.runner import run_langgraph
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
        graph_run, "settings", Settings.model_construct(ENABLE_LANGFUSE=True)
    )
    monkeypatch.setattr(
        graph_run,
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

    message = await run_langgraph(
        request, [HumanMessage(content="question")], graph_registry
    )

    assert message.text == "hello"
    assert mock_langfuse_callback.starts == 1

    if recording_callback:
        assert recording_callback.starts == 1
        assert graph_config.runtime_callbacks == [recording_callback]
    else:
        assert graph_config.runtime_callbacks is None


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
        metadata={"conversation_id": "conversation-123"},
    )

    await run_langgraph(request, [HumanMessage(content="question")], graph_registry)

    assert recording_callback.root_metadata
    metadata = recording_callback.root_metadata[0]
    assert metadata["lgos.model"] == "interruptible"
    assert metadata["lgos.operation_id"]
    assert metadata["langfuse_session_id"] == "conversation-123"
    assert metadata["thread_id"]
    assert get_log_context() == {}


@pytest.mark.parametrize("conversation_id", [None, "", "conversation-123"])
async def test_callbacks_observe_request_correlation_metadata(
    make_request,
    conversation_id: str | None,
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
            **(
                {"conversation_id": conversation_id}
                if conversation_id is not None
                else {}
            ),
            "unrelated": "not callback metadata",
        },
    )
    token = begin_log_context("request-123")

    try:
        await run_langgraph(request, [HumanMessage(content="question")], graph_registry)
    finally:
        reset_log_context(token)

    metadata = recording_callback.root_metadata[0]
    assert metadata["lgos.model"] == "messages"
    assert metadata["lgos.request_id"] == "request-123"
    assert metadata.get("langfuse_session_id") == (conversation_id or None)
    assert "unrelated" not in metadata


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


def test_standard_graph_rejects_interrupt_run_coordinator() -> None:
    with pytest.raises(ValidationError, match="interrupt-enabled"):
        GraphConfig(
            graph=make_message_graph("hello"),
            description="DUMMY",
            run_coordinator=InMemoryRunCoordinator(),
        )


async def test_unknown_model_raises_model_not_found(make_request) -> None:
    request = make_request("missing")
    graph_registry = GraphRegistry(
        registry={
            "known": GraphConfig(
                graph=make_message_graph("hello"),
                description="DUMMY",
            ),
        }
    )

    with pytest.raises(InvalidRequestError, match="'missing' does not exist") as exc:
        await run_langgraph(request, [HumanMessage(content="question")], graph_registry)

    assert (exc.value.status_code, exc.value.code) == (404, "model_not_found")
