"""Describe gateway graph models the way Open WebUI presents models."""

import re
from collections.abc import Sequence
from typing import Annotated, TypedDict

from openai.types import Model
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StringConstraints,
    ValidationError,
)

from .contracts import (
    BACKGROUND_FEATURE,
    BACKGROUND_SETTING_NAME,
    FILE_INPUTS_FEATURE,
    LGOS_EXTENSION_KEY,
    MCP_TOOLS_FEATURE,
)

# Open WebUI's rule for Chat Variable keys.
CHAT_VARIABLE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
# The UserValves example reuses this graph with the uservalves_simple Filter.
USERVALVES_MODEL_ID = "uservalves_simple"
LIMITED_FUNCTIONALITY_DESCRIPTION = (
    "Limited functionality: the configured gateway did not return valid lgos "
    "model metadata. Runtime settings, file inputs, and gateway tools may be "
    "unavailable."
)
# Open WebUI also shows a control for background delivery, and one per server
# tool, in the Chat Variables form. The Pipe consumes them; they never become
# graph settings.
BACKGROUND_FIELD: dict[str, JsonValue] = {
    "key": BACKGROUND_SETTING_NAME,
    "type": "checkbox",
    "label": "Run in background",
    "default": False,
}


class _ModelExtension(BaseModel):
    """Read the LGOS envelope; unsupported settings need not hide the model."""

    model_config = ConfigDict(strict=True)

    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    features: list[str]
    server_tools: list[str]
    client_tools: list[str]
    client_settings: JsonValue = None


class _ClientSettings(BaseModel):
    """Validate the settings envelope before projecting its supported fields."""

    model_config = ConfigDict(strict=True)

    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


class OpenWebUIModel(TypedDict):
    """One ``pipes()`` entry; the image keeps its ``info`` as the model's info."""

    id: str
    name: str
    info: dict[str, JsonValue]


def openwebui_models(
    models: Sequence[Model],
    *,
    mcp_gateway_connection_id: str,
    uservalves_graph_model_id: str | None,
) -> list[OpenWebUIModel]:
    """Describe gateway models, plus an explicitly configured demo example."""
    result: list[OpenWebUIModel] = []
    for model in sorted(models, key=lambda item: item.id):
        extension = _model_extension(model)
        meta = _model_meta(extension, mcp_gateway_connection_id)
        suffix = "" if extension is not None else " (Limited functionality)"
        result.append(
            {"id": model.id, "name": f"{model.id}{suffix}", "info": {"meta": meta}}
        )
        if model.id == uservalves_graph_model_id and extension is not None:
            # Its Filter supplies the settings instead of Chat Variables.
            uservalves_meta = {
                key: value
                for key, value in meta.items()
                if key != "chat_variables_schema"
            }
            uservalves_meta.update(
                description="Static per-user history and audience settings.",
                filterIds=[USERVALVES_MODEL_ID],
            )
            result.append(
                {
                    "id": USERVALVES_MODEL_ID,
                    "name": "UserValves Simple / simple-graph",
                    "info": {"meta": uservalves_meta},
                }
            )
    return result


def _model_meta(
    extension: _ModelExtension | None, mcp_gateway_connection_id: str
) -> dict[str, JsonValue]:
    features = extension.features if extension is not None else []
    server_tools = extension.server_tools if extension is not None else []
    fields: list[JsonValue] = [
        *_chat_variable_fields(extension),
        *(
            _server_tool_field(name)
            for name in server_tools
            if CHAT_VARIABLE_KEY.fullmatch(name)
        ),
    ]
    if BACKGROUND_FEATURE in features:
        fields.append(BACKGROUND_FIELD)
    meta: dict[str, JsonValue] = {
        "description": (
            extension.description
            if extension is not None
            else LIMITED_FUNCTIONALITY_DESCRIPTION
        ),
        "capabilities": {
            "file_upload": FILE_INPUTS_FEATURE in features,
            "file_context": False,
        },
        "builtinTools": {"files": False},
    }
    if MCP_TOOLS_FEATURE in features:
        meta["toolIds"] = [f"server:mcp:{mcp_gateway_connection_id}"]
    if extension is not None:
        # The Pipe reads these back from the selected model at chat time.
        meta[LGOS_EXTENSION_KEY] = {
            "server_tools": list(extension.server_tools),
            "client_tools": list(extension.client_tools),
        }
    if fields:
        meta["chat_variables_schema"] = {"fields": fields}
    return meta


def _server_tool_field(name: str) -> dict[str, JsonValue]:
    """Offer one server tool as an unchecked box labelled from its name."""
    return {
        "key": name,
        "type": "checkbox",
        "label": name.replace("_", " ").capitalize(),
        "default": False,
    }


def _model_extension(model: Model) -> _ModelExtension | None:
    extension = (model.model_extra or {}).get(LGOS_EXTENSION_KEY)
    try:
        return _ModelExtension.model_validate(extension)
    except ValidationError:
        return None


def _chat_variable_fields(
    extension: _ModelExtension | None,
) -> tuple[dict[str, JsonValue], ...]:
    """Translate the Chainlit-supported LGOS schema subset to Chat Variables."""
    if extension is None or extension.client_settings is None:
        return ()

    try:
        settings = _ClientSettings.model_validate(extension.client_settings)
    except ValidationError:
        return ()

    properties = settings.json_schema.get("properties")
    if not isinstance(properties, dict):
        return ()

    fields = []
    for name, default in settings.defaults.items():
        field = _chat_variable_field(name, properties.get(name), default)
        if field is not None:
            fields.append(field)
    return tuple(fields)


def _chat_variable_field(  # ruff: ignore[too-many-return-statements] - Each supported schema shape returns its own field declaration.
    name: str,
    schema: JsonValue,
    default: JsonValue,
) -> dict[str, JsonValue] | None:
    if CHAT_VARIABLE_KEY.fullmatch(name) is None or not isinstance(schema, dict):
        return None

    title = schema.get("title")
    field: dict[str, JsonValue] = {
        "key": name,
        "label": (
            title
            if isinstance(title, str) and title
            else name.replace("_", " ").title()
        ),
        "default": default,
    }
    schema_type = schema.get("type")
    if schema_type == "boolean" and type(default) is bool:
        return {**field, "type": "checkbox"}
    if schema_type == "integer" and type(default) is int:
        field = {**field, "type": "number", "step": 1}
        if type(schema.get("minimum")) is int:
            field["min"] = schema["minimum"]
        if type(schema.get("maximum")) is int:
            field["max"] = schema["maximum"]
        return field
    if schema_type != "string" or not isinstance(default, str):
        return None

    enum = schema.get("enum")
    if enum is None:
        return {**field, "type": "text"}
    if (
        not isinstance(enum, list)
        or any(not isinstance(value, str) for value in enum)
        or len(set(enum)) != len(enum)
        or default not in enum
    ):
        return None
    return {**field, "type": "select", "options": enum}
