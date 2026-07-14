from typing import Literal

from pydantic import BaseModel

from langgraph_openai_serve.graph.features import GraphFeature


class ModelPermission(BaseModel):
    """Model permission information."""

    id: str
    object: str = "model_permission"
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: str | None = None
    is_blocking: bool


class LangGraphModelExtension(BaseModel):
    """Versioned LangGraph OpenAI Serve model extension."""

    schema_version: Literal[1] = 1
    features: list[GraphFeature]


class Model(BaseModel):
    """Individual model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission]
    langgraph_openai_serve: LangGraphModelExtension


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[Model]
