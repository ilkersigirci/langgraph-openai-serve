"""Public graph settings transported through standard OpenAI requests."""

from functools import cache
from typing import NamedTuple, Self, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest

RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"


class ClientSettings(BaseModel):
    """Base class for settings that clients may configure for a graph.

    Subclasses define the complete public contract. Every field must have a
    valid JSON-serializable default so model discovery can advertise a usable
    settings object without maintaining a second defaults mapping.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    @classmethod
    def defaults(cls) -> Self:
        """Return a deep copy of the registration-validated defaults."""
        return cast(
            Self,
            _validated_contract(cls).defaults.model_copy(deep=True),
        )

    @classmethod
    def validate_request(cls, request: ChatCompletionRequest) -> Self:
        """Read and validate this model's values from an OpenAI request."""
        parameter = f"metadata.{RUNTIME_SETTINGS_METADATA_KEY}"
        encoded = (request.metadata or {}).get(RUNTIME_SETTINGS_METADATA_KEY, "{}")

        try:
            changes = _validate_json_object(encoded)
        except ValidationError as exc:
            raise ClientSettingsValidationError(
                _validation_message(exc, label="runtime settings"),
                param=parameter,
            ) from exc

        values = client_settings_default_values(cls)
        values.update(changes)

        try:
            settings = cls.model_validate_json(
                _SETTINGS_OBJECT_ADAPTER.dump_json(values),
                strict=True,
                by_alias=False,
                by_name=True,
            )
            _validated_settings_json(settings)
            return settings
        except ValidationError as exc:
            field = _first_error_field(exc)
            raise ClientSettingsValidationError(
                _validation_message(exc, label=field or "runtime settings"),
                param=parameter,
            ) from exc


class ClientSettingsValidationError(ValueError):
    """Raised when client-controlled graph settings are invalid."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


class _ValidatedContract(NamedTuple):
    defaults: ClientSettings
    defaults_json: bytes
    json_schema_json: bytes


_SETTINGS_OBJECT_ADAPTER = TypeAdapter(
    dict[str, JsonValue],
    config=ConfigDict(allow_inf_nan=False),
)


def client_settings_default_values(
    settings_model: type[ClientSettings],
) -> dict[str, JsonValue]:
    """Return a fresh copy of the registration-validated JSON defaults."""
    return _validate_json_object(_validated_contract(settings_model).defaults_json)


def client_settings_json_schema(
    settings_model: type[ClientSettings],
) -> dict[str, JsonValue]:
    """Return a fresh copy of the registration-validated discovery schema."""
    return _validate_json_object(_validated_contract(settings_model).json_schema_json)


@cache
def _validated_contract(
    settings_model: type[ClientSettings],
) -> _ValidatedContract:
    """Freeze validated defaults and discovery schema at registration."""
    # by_alias/by_name validation overrides require Pydantic 2.11 or newer.
    default = settings_model.model_validate_json(
        "{}",
        strict=True,
        by_alias=False,
        by_name=True,
    )
    encoded = _validated_settings_json(default)
    validated = settings_model.model_validate_json(
        encoded,
        strict=True,
        by_alias=False,
        by_name=True,
    )
    defaults_json = _validated_settings_json(validated)
    schema = _SETTINGS_OBJECT_ADAPTER.validate_python(
        settings_model.model_json_schema(by_alias=False),
        strict=True,
    )
    return _ValidatedContract(
        defaults=validated,
        defaults_json=defaults_json,
        json_schema_json=_SETTINGS_OBJECT_ADAPTER.dump_json(
            schema,
            warnings="error",
        ),
    )


def validate_client_settings_model(
    settings_model: type[ClientSettings],
) -> type[ClientSettings]:
    """Validate the registration-time contract of a settings model."""
    if any(
        settings_model.model_config.get(key) != expected
        for key, expected in ClientSettings.model_config.items()
    ):
        raise ValueError(
            "ClientSettings subclasses must preserve the inherited model config."
        )

    if any(
        field.exclude or getattr(field, "exclude_if", None) is not None
        for field in settings_model.model_fields.values()
    ):
        raise ValueError("ClientSettings fields cannot be excluded from defaults.")

    client_settings_json_schema(settings_model)
    return settings_model


def _validate_json_object(encoded: str | bytes) -> dict[str, JsonValue]:
    """Decode a JSON object, then reject non-finite decoded numbers."""
    value = _SETTINGS_OBJECT_ADAPTER.validate_json(encoded, strict=True)
    return _SETTINGS_OBJECT_ADAPTER.validate_python(value, strict=True)


def _validated_settings_json(settings: ClientSettings) -> bytes:
    """Serialize settings only after checking their final JSON values."""
    values = settings.model_dump(mode="json", by_alias=False, warnings="error")
    validated = _SETTINGS_OBJECT_ADAPTER.validate_python(values, strict=True)
    return _SETTINGS_OBJECT_ADAPTER.dump_json(validated, warnings="error")


def _first_error_field(error: ValidationError) -> str | None:
    errors = error.errors()
    location = errors[0].get("loc", ()) if errors else ()
    return str(location[0]) if location else None


def _validation_message(error: ValidationError, *, label: str) -> str:
    errors = error.errors()
    message = str(errors[0].get("msg") or "Invalid value") if errors else str(error)
    if label == "runtime settings":
        return f"Invalid runtime settings: {message}"
    return f"Invalid runtime setting for {label}: {message}"
