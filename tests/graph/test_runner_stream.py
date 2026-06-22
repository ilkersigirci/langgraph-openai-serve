import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.constants import TAG_HIDDEN
from langgraph.graph import StateGraph

from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph_stream
from tests.graph.conftest import (
    AnswerOutput,
    MessageState,
    QuestionInput,
    QuestionState,
)


async def stream_text(name: str, graph_registry: GraphRegistry, make_request) -> str:
    chat_request = make_request(name)
    chunks = run_langgraph_stream(
        name,
        chat_request.messages,
        graph_registry,
        chat_request,
    )
    return "".join([chunk async for chunk, _ in chunks])


@pytest.mark.anyio
async def test_nested_subgraph_streaming(
    make_request,
) -> None:
    model = FakeListChatModel(responses=["nested"])

    async def generate(state: QuestionState):
        await model.ainvoke([HumanMessage(content=state["question"])])
        return {"answer": "done"}

    subgraph = (
        StateGraph(QuestionState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    graph = (
        StateGraph(
            QuestionState, input_schema=QuestionInput, output_schema=AnswerOutput
        )
        .add_node("subgraph", subgraph)
        .set_entry_point("subgraph")
        .set_finish_point("subgraph")
        .compile()
    )
    graph_registry = GraphRegistry(
        registry={
            "nested": GraphConfig(
                graph=graph,
                request_to_input=lambda request, messages: {
                    "question": messages[-1].content
                },
                output_to_text=lambda output: output["answer"],
                streamable_node_names=["generate"],
            )
        },
    )

    assert await stream_text("nested", graph_registry, make_request) == "nested"


@pytest.mark.anyio
async def test_stream_filters_nodes_hidden_tags_and_non_ai_messages(
    make_request,
) -> None:
    draft_model = FakeListChatModel(responses=["draft"])
    hidden_model = FakeListChatModel(responses=["hidden"]).with_config(
        tags=[TAG_HIDDEN]
    )
    visible_model = FakeListChatModel(responses=["visible"])

    async def draft(state: MessageState):
        return {"messages": [await draft_model.ainvoke(state["messages"])]}

    async def generate(state: MessageState):
        await hidden_model.ainvoke(state["messages"])
        return {"messages": [await visible_model.ainvoke(state["messages"])]}

    builder = StateGraph(MessageState)
    builder.add_node(
        "non_ai",
        lambda state: {"messages": [HumanMessage(content="ignored")]},
    )
    builder.add_node("draft", draft)
    builder.add_node("generate", generate)
    builder.set_entry_point("non_ai")
    builder.add_edge("non_ai", "draft")
    builder.add_edge("draft", "generate")
    builder.set_finish_point("generate")

    graph_registry = GraphRegistry(
        registry={
            "filtered": GraphConfig(
                graph=builder.compile(),
                streamable_node_names=["non_ai", "generate"],
            )
        },
    )

    assert await stream_text("filtered", graph_registry, make_request) == "visible"
