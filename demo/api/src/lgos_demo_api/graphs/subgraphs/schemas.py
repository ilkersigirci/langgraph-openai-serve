"""Pydantic state schemas shared by the complex subgraph demo."""

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["api", "docs"]


class ComplexSubgraphState(BaseModel):
    question: str = ""
    normalized_question: str = ""
    route: Route = "docs"
    answer: str = ""


class ApiContractState(BaseModel):
    question: str = ""
    normalized_question: str = ""
    checks: list[str] = Field(default_factory=list)
    answer: str = ""


class DocsState(BaseModel):
    question: str = ""
    normalized_question: str = ""
    keywords: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    answer: str = ""


class KeywordState(BaseModel):
    question: str = ""
    normalized_question: str = ""
    keywords: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
