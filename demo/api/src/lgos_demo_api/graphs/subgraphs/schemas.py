"""Pydantic state schemas shared by the complex subgraph demo."""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["api", "docs"]


class MessageState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)


class ComplexSubgraphState(MessageState):
    question: str = ""
    normalized_question: str = ""
    route: Route = "docs"


class ApiContractState(MessageState):
    question: str = ""
    normalized_question: str = ""
    checks: list[str] = Field(default_factory=list)


class DocsState(MessageState):
    question: str = ""
    normalized_question: str = ""
    keywords: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class KeywordState(MessageState):
    question: str = ""
    normalized_question: str = ""
    keywords: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
