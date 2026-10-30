import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from langgraph_openai_serve.core.errors import InvalidRequestError
from langgraph_openai_serve.core.logging import (
    begin_log_context,
    get_log_context,
    reset_log_context,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.runner import run_langgraph
from tests.graph.support.interrupt import make_interrupt_graph
from tests.graph.support.message import make_message_graph


class RecordingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.root_metadata: list[dict[str, object]] = []

    def on_chain_start(
        self,
        *args,
        parent_run_id=None,
        metadata=None,
        **kwargs,
    ) -> None:
        if parent_run_id is None:
            self.root_metadata.append(dict(metadata or {}))


async def test_interrupt_callback_observes_native_checkpoint_metadata(
    make_request,
    sqlite_checkpointer,
) -> None:
    recording_callback = RecordingCallback()
    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer).with_config(
            callbacks=[recording_callback]
        ),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
    )
    graph_registry = GraphRegistry(
        graphs={"interruptible": graph_config},
        run_coordinator=InMemoryRunCoordinator(),
    )
    request = make_request("interruptible")

    await run_langgraph(request, [HumanMessage(content="question")], graph_registry)

    assert recording_callback.root_metadata
    metadata = recording_callback.root_metadata[0]
    assert metadata["lgos.model"] == "interruptible"
    assert metadata["lgos.operation_id"]
    assert metadata["thread_id"]
    assert get_log_context() == {}


@pytest.mark.parametrize("user", [None, "", "user-123"])
async def test_callbacks_observe_request_correlation_metadata(
    make_request,
    user: str | None,
) -> None:
    recording_callback = RecordingCallback()
    graph_config = GraphConfig(
        graph=make_message_graph("hello").with_config(callbacks=[recording_callback]),
        description="DUMMY",
    )
    graph_registry = GraphRegistry(graphs={"messages": graph_config})
    request = make_request(
        "messages",
        user=user,
        metadata={
            "conversation_id": "conversation-123",
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
    assert metadata.get("lgos.user") == (user or None)
    # Request metadata stays out of callback metadata; Langfuse receives the
    # conversation as its native session instead.
    assert "conversation-123" not in metadata.values()
    assert "unrelated" not in metadata


async def test_unknown_model_raises_model_not_found(make_request) -> None:
    request = make_request("missing")
    graph_registry = GraphRegistry(
        graphs={
            "known": GraphConfig(
                graph=make_message_graph("hello"),
                description="DUMMY",
            ),
        }
    )

    with pytest.raises(InvalidRequestError, match="'missing' does not exist") as exc:
        await run_langgraph(request, [HumanMessage(content="question")], graph_registry)

    assert (exc.value.status_code, exc.value.code) == (404, "model_not_found")
