import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import CustomStreamPart

from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphNotFoundError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.runner import (
    invoke_run,
    run_langgraph,
)
from langgraph_openai_serve.graph.utils import GraphRun
from tests.graph.support.message import make_message_graph

pytestmark = pytest.mark.anyio


class RecordingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.starts += 1


async def test_runtime_callbacks_reach_default_message_graph(
    make_request,
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

    invocation = await run_langgraph(
        "messages",
        chat_request.messages,
        graph_registry,
        chat_request,
    )

    assert invocation.output == "hello"
    assert recording_callback.starts == 1


async def test_unknown_model_raises_graph_not_found_error(make_request) -> None:
    chat_request = make_request("missing")
    graph_registry = GraphRegistry(
        registry={
            "known": GraphConfig(graph=make_message_graph("hello")),
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
            output_to_text=lambda output: output["answer"],
        ),
        graph=graph,
        inputs={},
        context=None,
        runnable_config=None,
        thread_id=None,
    )

    invocation = await invoke_run(run)

    assert invocation.output == "done"
    assert invocation.custom_events == (
        CustomStreamPart(type="custom", ns=(), data=payload),
    )
