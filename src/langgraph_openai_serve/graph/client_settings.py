"""Public graph settings transported through standard OpenAI requests."""

from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from langgraph_openai_serve.core.errors import InvalidRequestError
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.protocol import (
    JSON_SCHEMA_DIALECT,
    SETTINGS_METADATA_KEY,
)


class ClientSettings(BaseModel):
    """
    Base class for settings that clients may configure for a graph.

    Subclasses define the complete public contract. Every field needs a default
    so model discovery can advertise a complete settings object.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    @classmethod
    def validate_request(cls, request: GraphRequest) -> Self:
        """Read and validate this model's values from an OpenAI request."""
        try:
            return cls.model_validate_json(
                request.metadata.get(SETTINGS_METADATA_KEY, "{}")
            )
        except ValidationError as exc:
            error = exc.errors()[0]
            field = ".".join(str(part) for part in error["loc"])
            label = f"runtime setting for {field}" if field else "runtime settings"
            msg = f"Invalid {label}: {error['msg']}"
            raise InvalidRequestError(
                msg,
                param=f"metadata.{SETTINGS_METADATA_KEY}",
            ) from exc

    @classmethod
    def json_schema(cls) -> dict[str, JsonValue]:
        """Return the JSON Schema advertised by model discovery."""
        return {"$schema": JSON_SCHEMA_DIALECT, **cls.model_json_schema()}

    @classmethod
    def default_values(cls) -> dict[str, JsonValue]:
        """Return the JSON defaults advertised by model discovery."""
        return cls().model_dump(mode="json")
