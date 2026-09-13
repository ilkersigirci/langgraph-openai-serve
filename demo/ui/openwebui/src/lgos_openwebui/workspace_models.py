"""Generate Open WebUI Workspace Models from LGOS model metadata."""

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import httpx
from openai import OpenAI, OpenAIError
from openai.types import Model
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StringConstraints,
    ValidationError,
)

from .functions.generic.api import _model_request
from .functions.generic.contracts import (
    CURRENT_TIME_TOOL_NAME,
    LGOS_EXTENSION_KEY,
    LGOS_MODEL_OWNER,
    WEB_SEARCH_TOOL_NAME,
    is_server_tool_model,
)
from .functions.generic.gateway import GatewayConfig, litellm_models

FILE_INPUTS_FEATURE = "file_inputs"
CHAT_VARIABLES_META_KEY = "chat_variables_schema"
CHAT_VARIABLE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
GENERIC_FUNCTION_ID = "generic"
WORKSPACE_MODEL_PREFIX = "lgos."
USERVALVES_MODEL_ID = "lgos.uservalves_simple"
OPENWEBUI_MODEL_ID_MAX_LENGTH = 256
PUBLIC_READ_GRANT = {
    "principal_type": "user",
    "principal_id": "*",
    "permission": "read",
}
LIMITED_FUNCTIONALITY_DESCRIPTION = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "lgos model metadata. Runtime settings, file inputs, and "
    "interrupt profile checks may be unavailable."
)
SERVER_TOOL_FIELDS: tuple[dict[str, JsonValue], ...] = (
    {
        "key": CURRENT_TIME_TOOL_NAME,
        "type": "checkbox",
        "label": "Current time",
        "default": False,
    },
    {
        "key": WEB_SEARCH_TOOL_NAME,
        "type": "checkbox",
        "label": "Web search",
        "default": False,
    },
)


class _ModelExtension(BaseModel):
    """Read the LGOS envelope; unsupported settings need not hide the model."""

    model_config = ConfigDict(strict=True)

    schema_version: Literal[1]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    features: list[str]
    client_settings: JsonValue = None


class _ClientSettings(BaseModel):
    """Validate the settings envelope before projecting its supported fields."""

    model_config = ConfigDict(strict=True)

    schema_version: Literal[1]
    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


@dataclass(frozen=True)
class WorkspaceModelSpec:
    """Describe one generated Open WebUI Workspace Model."""

    id: str
    fields: tuple[dict[str, JsonValue], ...]
    description: str | None = None
    supports_file_inputs: bool = False

    def __post_init__(self) -> None:
        if len(self.base_model_id) > OPENWEBUI_MODEL_ID_MAX_LENGTH:
            msg = f"LGOS model ID is too long for Open WebUI: {self.id}"
            raise ValueError(msg)

    @property
    def limited(self) -> bool:
        return self.description is None

    @property
    def name(self) -> str:
        suffix = " (Limited functionality)" if self.limited else ""
        return f"LGOS / {self.id}{suffix}"

    @property
    def workspace_model_id(self) -> str:
        return f"{WORKSPACE_MODEL_PREFIX}{self.id}"

    @property
    def base_model_id(self) -> str:
        return f"{GENERIC_FUNCTION_ID}.{self.id}"


def chat_variable_fields(model: Model) -> tuple[dict[str, JsonValue], ...] | None:
    """Translate the Chainlit-supported LGOS schema subset to Chat Variables."""
    extension = _model_extension(model)
    return _chat_variable_fields(extension) if extension is not None else None


def _chat_variable_fields(
    extension: _ModelExtension | None,
) -> tuple[dict[str, JsonValue], ...]:
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


def discover_workspace_model_specs(
    client: OpenAI,
    *,
    gateway: GatewayConfig,
) -> tuple[WorkspaceModelSpec, ...]:
    """Build Workspace Models from the configured OpenAI model endpoints."""
    models: dict[str, Model | None] = {}
    if not gateway.provider_routing:
        payload = client.get(f"{gateway.root_url}/model/info", cast_to=object)
        models = {model.id: model for model in litellm_models(payload)}
    else:
        catalog = client.with_options(base_url=f"{gateway.root_url}/v1").models.list()
        detail_client = client.with_options(
            base_url=f"{gateway.root_url}/openai_passthrough/v1"
        )
        for catalog_model in catalog.data:
            if catalog_model.owned_by != LGOS_MODEL_OWNER:
                continue
            try:
                models[catalog_model.id] = detail_client.models.retrieve(
                    **_model_request(catalog_model.id, provider_routing=True)
                )
            except OpenAIError:
                models[catalog_model.id] = None

    specs = []
    for model_id, model in sorted(models.items()):
        extension = _model_extension(model)
        specs.append(
            WorkspaceModelSpec(
                id=model_id,
                fields=_chat_variable_fields(extension),
                description=extension.description if extension is not None else None,
                supports_file_inputs=(
                    extension is not None and FILE_INPUTS_FEATURE in extension.features
                ),
            )
        )
    return tuple(specs)


def sync_workspace_models(
    client: httpx.Client,
    specs: tuple[WorkspaceModelSpec, ...],
) -> None:
    """Replace generated Workspace Models and their hidden manifold bases."""
    workspace_models = client.get("/api/v1/models/export").raise_for_status().json()
    if not isinstance(workspace_models, list):
        msg = "Open WebUI models export returned invalid data."
        raise TypeError(msg)
    base_models = client.get("/api/v1/models/base").raise_for_status().json()
    if not isinstance(base_models, list):
        msg = "Open WebUI base models response returned invalid data."
        raise TypeError(msg)
    existing_model_ids = {
        model["id"]
        for model in workspace_models
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    }
    desired_base_model_ids = {spec.base_model_id for spec in specs}
    generated_workspace_model_ids = {
        model["id"]
        for model in workspace_models
        if isinstance(model, dict)
        and isinstance(model.get("id"), str)
        and model["id"].startswith(WORKSPACE_MODEL_PREFIX)
        and isinstance(model.get("base_model_id"), str)
        and model["base_model_id"].startswith(f"{GENERIC_FUNCTION_ID}.")
    }
    generated_base_model_ids = {
        model["id"]
        for model in base_models
        if isinstance(model, dict)
        and isinstance(model.get("id"), str)
        and model["id"].startswith(f"{GENERIC_FUNCTION_ID}.")
        and model.get("base_model_id") is None
    }

    payloads = []
    for spec in specs:
        payloads.append(_hidden_base_model_payload(spec))
        payloads.append(_workspace_model_payload(spec))
        if spec.id == "lgos-a/simple-graph" and not spec.limited:
            simple_model = _workspace_model_payload(spec)
            simple_model.update(
                id=USERVALVES_MODEL_ID,
                name="UserValves Simple / simple-graph",
            )
            simple_model["meta"].update(
                description="Static per-user history and audience settings.",
                chat_variables_schema={"fields": []},
                filterIds=["uservalves_simple"],
            )
            payloads.append(simple_model)

    desired_workspace_model_ids = {
        model["id"] for model in payloads if model["base_model_id"] is not None
    }
    for workspace_model in payloads:
        # Open WebUI preserves existing grants when imports omit this field.
        if workspace_model["id"] not in existing_model_ids:
            workspace_model["access_grants"] = [PUBLIC_READ_GRANT]

    if payloads:
        client.post(
            "/api/v1/models/import",
            json={"models": payloads},
        ).raise_for_status()

    stale_workspace_model_ids = (
        generated_workspace_model_ids - desired_workspace_model_ids
    )
    stale_base_model_ids = generated_base_model_ids - desired_base_model_ids
    for model_id in sorted(stale_workspace_model_ids):
        client.post(
            "/api/v1/models/model/delete",
            json={"id": model_id},
        ).raise_for_status()
    for model_id in sorted(stale_base_model_ids):
        client.post(
            "/api/v1/models/model/delete",
            json={"id": model_id},
        ).raise_for_status()


def _chat_variable_field(
    name: str,
    schema: JsonValue,
    default: JsonValue,
) -> dict[str, JsonValue] | None:
    if CHAT_VARIABLE_KEY.fullmatch(name) is None or not isinstance(schema, dict):
        return None

    field: dict[str, JsonValue] = {
        "key": name,
        "label": str(schema.get("title") or name.replace("_", " ").title()),
        "default": default,
    }
    schema_type = schema.get("type")
    if schema_type == "boolean" and type(default) is bool:
        return {**field, "type": "checkbox"}
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


def _model_extension(model: Model | None) -> _ModelExtension | None:
    extension = (model.model_extra or {}).get(LGOS_EXTENSION_KEY) if model else None
    try:
        return _ModelExtension.model_validate(extension)
    except ValidationError:
        return None


def _hidden_base_model_payload(spec: WorkspaceModelSpec) -> dict[str, Any]:
    return {
        "id": spec.base_model_id,
        "base_model_id": None,
        "name": f"Generic / {spec.id}",
        "meta": {"hidden": True},
        "params": {},
        "access_grants": [PUBLIC_READ_GRANT],
        "is_active": True,
    }


def _workspace_model_payload(spec: WorkspaceModelSpec) -> dict[str, Any]:
    # Open WebUI reads this native schema from Workspace Model metadata.
    # Keeping it out of params.system prevents settings UI data from becoming
    # an LGOS system prompt.
    fields = list(spec.fields)
    if is_server_tool_model(spec.id):
        fields.extend(SERVER_TOOL_FIELDS)
    return {
        "id": spec.workspace_model_id,
        "base_model_id": spec.base_model_id,
        "name": spec.name,
        "meta": {
            "description": spec.description or LIMITED_FUNCTIONALITY_DESCRIPTION,
            CHAT_VARIABLES_META_KEY: {"fields": fields},
            "capabilities": {
                "file_upload": spec.supports_file_inputs,
                "file_context": False,
            },
            "builtinTools": {"files": False},
        },
        "params": {},
    }
