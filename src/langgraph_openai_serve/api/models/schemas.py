from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints

from langgraph_openai_serve.graph.features import GraphFeature


class ModelClientSettings(BaseModel):
    """Versioned public runtime settings for one registered graph."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1] = 1
    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


class LangGraphModelSummaryExtension(BaseModel):
    """Versioned LGOS fields safe to include in a model list."""

    schema_version: Literal[1] = 1
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]


class LangGraphModelExtension(LangGraphModelSummaryExtension):
    """Versioned LangGraph OpenAI Serve model-detail extension."""

    features: list[GraphFeature]
    client_settings: ModelClientSettings | None = None


class Model(BaseModel):
    """Individual model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    langgraph_openai_serve: LangGraphModelSummaryExtension


class ModelDetails(Model):
    """Retrieved model with required LGOS capability metadata."""

    langgraph_openai_serve: LangGraphModelExtension


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[Model]
