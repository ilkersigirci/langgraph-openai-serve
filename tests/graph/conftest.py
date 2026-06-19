from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import BaseMessage
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
