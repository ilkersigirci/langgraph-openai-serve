from collections.abc import Callable
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest, Role


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class QuestionState(TypedDict, total=False):
    question: str
    answer: str
    internal: str


class QuestionInput(TypedDict):
    question: str


class AnswerOutput(TypedDict):
    answer: str


class PydanticQuestionState(BaseModel):
    question: str
    answer: str = ""


class PydanticQuestionInput(BaseModel):
    question: str


class PydanticAnswerOutput(BaseModel):
    answer: str


@pytest.fixture
def make_request() -> Callable[..., ChatCompletionRequest]:
    def _make_request(
        model: str,
        *,
        content: str = "question",
        user: str | None = None,
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=model,
            messages=[{"role": Role.USER, "content": content}],
            user=user,
        )

    return _make_request


@pytest.fixture
def make_message_graph() -> Callable[..., Any]:
    def _make_message_graph(response: str = "hello", *, node_name: str = "generate"):
        model = FakeListChatModel(responses=[response])

        async def generate(state: MessageState):
            return {"messages": [await model.ainvoke(state["messages"])]}

        return (
            StateGraph(MessageState)
            .add_node(node_name, generate)
            .set_entry_point(node_name)
            .set_finish_point(node_name)
            .compile()
        )

    return _make_message_graph
