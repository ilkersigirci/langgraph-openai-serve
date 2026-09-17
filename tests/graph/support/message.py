from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import StateGraph

from tests.graph.support.schemas import MessageState


def make_message_graph(
    response: str = "hello",
    *,
    node_name: str = "generate",
    context_schema: type[Any] | None = None,
    disable_streaming: bool = False,
) -> Any:
    model = FakeListChatModel(
        responses=[response],
        disable_streaming=disable_streaming,
    )

    async def generate(state: MessageState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    return (
        StateGraph(MessageState, context_schema=context_schema)
        .add_node(node_name, generate)
        .set_entry_point(node_name)
        .set_finish_point(node_name)
        .compile()
    )
