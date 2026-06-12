from typing import Annotated, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@pytest.fixture
def message_graph():
    model = FakeListChatModel(responses=["hello"])

    async def generate(state: MessageState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    return (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
