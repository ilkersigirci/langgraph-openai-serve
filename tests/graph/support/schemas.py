from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


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
