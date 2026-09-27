from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints

from langgraph_openai_serve.graph.features import GraphFeature


class ModelClientSettings(BaseModel):
    """Public runtime settings for one registered graph."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


class LangGraphModelSummaryExtension(BaseModel):
    """LGOS fields safe to include in a model list."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    features: list[GraphFeature]


class LangGraphModelExtension(LangGraphModelSummaryExtension):
    """LangGraph OpenAI Serve model-detail extension."""

    client_settings: ModelClientSettings | None = None


class Model(BaseModel):
    """Individual model information."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    lgos: LangGraphModelSummaryExtension


class ModelDetails(Model):
    """Retrieved model with required LGOS capability metadata."""

    lgos: LangGraphModelExtension


class ModelList(BaseModel):
    """List of available models."""

    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[Model]
