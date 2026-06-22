import pytest
from langchain_core.callbacks import BaseCallbackHandler

from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph


class RecordingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.starts += 1


@pytest.mark.anyio
async def test_default_message_input_callbacks_and_usage(
    make_request,
    make_message_graph,
) -> None:
    recording_callback = RecordingCallback()
    graph_registry = GraphRegistry(
        registry={
            "messages": GraphConfig(
                graph=make_message_graph("hello"),
                runtime_callbacks=[recording_callback],
            )
        },
    )
    chat_request = make_request("messages")

    response, usage = await run_langgraph(
        "messages",
        chat_request.messages,
        graph_registry,
        chat_request,
    )

    assert response == "hello"
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert recording_callback.starts == 1


@pytest.mark.anyio
async def test_unknown_model_raises_value_error(make_request) -> None:
    chat_request = make_request("missing")

    with pytest.raises(ValueError, match="Graph 'missing' not found"):
        await run_langgraph(
            "missing",
            chat_request.messages,
            GraphRegistry(registry={}),
            chat_request,
        )
