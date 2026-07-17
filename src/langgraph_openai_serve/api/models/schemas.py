from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from langgraph_openai_serve.graph.features import GraphFeature


class ModelClientSettings(BaseModel):
    """Versioned public runtime settings for one registered graph."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


class LangGraphModelExtension(BaseModel):
    """Versioned LangGraph OpenAI Serve model extension."""

    schema_version: Literal[1] = 1
    features: list[GraphFeature]
    client_settings: ModelClientSettings | None = None


class Model(BaseModel):
    """Individual model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelDetails(Model):
    """Retrieved model with optional LGOS discovery metadata."""

    langgraph_openai_serve: LangGraphModelExtension


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[Model]
