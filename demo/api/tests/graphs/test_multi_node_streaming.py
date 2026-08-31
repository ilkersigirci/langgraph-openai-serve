from langchain_core.messages import AIMessage
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

from lgos_demo_api.graphs.multi_node_streaming import (
    ANSWER,
    multi_node_streaming_graph_config,
)


async def test_multiple_nodes_produce_the_same_streamed_and_complete_output(
    make_request,
) -> None:
    request = make_request(
        "multi-node-streaming",
        content="Build one answer from two nodes.",
    )
    registry = GraphRegistry(
        registry={"multi-node-streaming": multi_node_streaming_graph_config}
    )

    events = [
        event
        async for event in run_langgraph_stream(
            request.model,
            request.messages,
            registry,
            request,
        )
    ]
    complete = await run_langgraph(
        request.model,
        request.messages,
        registry,
        request,
    )

    streamed = "".join(event for event in events if isinstance(event, str))
    final_stream_message = events[-1]
    assert isinstance(final_stream_message, AIMessage)
    assert isinstance(complete.output, AIMessage)
    assert streamed == final_stream_message.text == complete.output.text == ANSWER
