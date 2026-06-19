import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import StateGraph

from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph
from tests.graph.conftest import MessageState


class RecordingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.starts += 1


@pytest.mark.anyio
async def test_message_defaults_callback_list_and_usage(
    make_request,
) -> None:
    model = FakeListChatModel(responses=["hello"])

    async def generate(state: MessageState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    recording_callback = RecordingCallback()
    graph_registry = GraphRegistry(
        registry={
            "messages": GraphConfig(
                graph=graph,
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
    assert usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }
    assert recording_callback.starts == 1


@pytest.mark.anyio
async def test_unknown_graph_raises_value_error(make_request) -> None:
    chat_request = make_request("missing")

    with pytest.raises(ValueError, match="Graph 'missing' not found"):
        await run_langgraph(
            "missing",
            chat_request.messages,
            GraphRegistry(registry={}),
            chat_request,
        )
