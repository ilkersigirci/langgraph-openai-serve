"""Public graph settings transported through standard OpenAI requests."""

from functools import cache
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest

CLIENT_CONFIG_METADATA_KEY = "langgraph_config"


class ClientSettings(BaseModel):
    """Base class for settings that clients may configure for a graph.

    Subclasses define the complete public contract. Every field must have a
    valid JSON-serializable default so model discovery can advertise a usable
    configuration without maintaining a second defaults mapping.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    @classmethod
    def defaults(cls) -> Self:
        """Build and deeply validate the model's JSON defaults."""
        # by_alias/by_name validation overrides require Pydantic 2.11 or newer.
        return cls.model_validate_json(
            _validated_defaults_json(cls),
            strict=True,
            by_alias=False,
            by_name=True,
        )

    @classmethod
    def validate_request(cls, request: ChatCompletionRequest) -> Self:
        """Read and validate this model's values from an OpenAI request."""
        parameter = f"metadata.{CLIENT_CONFIG_METADATA_KEY}"
        encoded = (request.metadata or {}).get(CLIENT_CONFIG_METADATA_KEY, "{}")

        try:
            changes = _CONFIG_OBJECT_ADAPTER.validate_json(encoded, strict=True)
        except ValidationError as exc:
            raise ClientConfigValidationError(
                _validation_message(exc, label="client configuration"),
                param=parameter,
            ) from exc

        values = _CONFIG_OBJECT_ADAPTER.validate_json(
            _validated_defaults_json(cls),
            strict=True,
        )
        values.update(changes)

        try:
            return cls.model_validate_json(
                _CONFIG_OBJECT_ADAPTER.dump_json(values),
                strict=True,
                by_alias=False,
                by_name=True,
            )
        except ValidationError as exc:
            field = _first_error_field(exc)
            raise ClientConfigValidationError(
                _validation_message(exc, label=field or "client configuration"),
                param=parameter,
            ) from exc


class ClientConfigValidationError(ValueError):
    """Raised when client-controlled graph settings are invalid."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


_CONFIG_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


@cache
def _validated_defaults_json(settings_model: type[ClientSettings]) -> str:
    """Freeze one validated JSON baseline for discovery and omitted values."""
    default = settings_model.model_validate_json(
        "{}",
        strict=True,
        by_alias=False,
        by_name=True,
    )
    encoded = default.model_dump_json(by_alias=False, warnings="error")
    validated = settings_model.model_validate_json(
        encoded,
        strict=True,
        by_alias=False,
        by_name=True,
    )
    return validated.model_dump_json(by_alias=False, warnings="error")


def validate_client_settings_model(
    settings_model: type[ClientSettings],
) -> type[ClientSettings]:
    """Validate the registration-time contract of a settings model."""
    settings_model.defaults()
    return settings_model


def _first_error_field(error: ValidationError) -> str | None:
    errors = error.errors()
    location = errors[0].get("loc", ()) if errors else ()
    return str(location[0]) if location else None


def _validation_message(error: ValidationError, *, label: str) -> str:
    errors = error.errors()
    message = str(errors[0].get("msg") or "Invalid value") if errors else str(error)
    if label == "client configuration":
        return f"Invalid client configuration: {message}"
    return f"Invalid client configuration for {label}: {message}"
