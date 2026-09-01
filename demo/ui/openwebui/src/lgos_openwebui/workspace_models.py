"""Generate Open WebUI Workspace Models from LGOS model metadata."""

import re
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI, OpenAIError

LGOS_EXTENSION_KEY = "langgraph_openai_serve"
CHAT_VARIABLES_META_KEY = "chat_variables_schema"
CHAT_VARIABLE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
GENERIC_FUNCTION_ID = "generic"
WORKSPACE_MODEL_PREFIX = "lgos."
OPENWEBUI_MODEL_ID_MAX_LENGTH = 256
LGOS_MODEL_OWNER = "langgraph-openai-serve"
PUBLIC_READ_GRANT = {
    "principal_type": "user",
    "principal_id": "*",
    "permission": "read",
}
LIMITED_FUNCTIONALITY_DESCRIPTION = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "langgraph_openai_serve model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable. Configure the proxy to pass LGOS /v1 requests "
    "and responses through unchanged."
)


@dataclass(frozen=True)
class WorkspaceModelSpec:
    """Describe one generated Open WebUI Workspace Model."""

    id: str
    fields: tuple[dict[str, Any], ...]
    description: str | None = None

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


def chat_variable_fields(model: Any) -> tuple[dict[str, Any], ...] | None:
    """Translate the Chainlit-supported LGOS schema subset to Chat Variables."""
    return _chat_variable_fields(_model_extension(model))


def _chat_variable_fields(
    extension: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    if extension is None:
        return None

    settings = extension.get("client_settings")
    if settings is None:
        return ()
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        return ()

    schema = settings.get("json_schema")
    defaults = settings.get("defaults")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict) or not isinstance(defaults, dict):
        return ()

    fields = []
    for name, default in defaults.items():
        field = _chat_variable_field(name, properties.get(name), default)
        if field is not None:
            fields.append(field)
    return tuple(fields)


def discover_workspace_model_specs(
    catalog_client: OpenAI,
    passthrough_client: OpenAI,
) -> tuple[WorkspaceModelSpec, ...]:
    """Build Workspace Models from Bifrost catalog and pass-through metadata."""
    specs = []
    for model_id in _list_model_ids(catalog_client):
        try:
            model = passthrough_client.models.retrieve(**_model_request(model_id))
        except OpenAIError:
            model = None
        extension = _model_extension(model)
        fields = _chat_variable_fields(extension)
        specs.append(
            WorkspaceModelSpec(
                id=model_id,
                fields=fields or (),
                description=(
                    extension["description"].strip() if extension is not None else None
                ),
            )
        )
    return tuple(sorted(specs, key=lambda spec: spec.id))


def _list_model_ids(client: OpenAI) -> list[str]:
    return [
        model.id
        for model in client.models.list().data
        if model.owned_by == LGOS_MODEL_OWNER
    ]


def _model_request(model_id: str) -> dict[str, Any]:
    provider, separator, upstream_model = model_id.partition("/")
    if not provider or not separator or not upstream_model:
        msg = f"Bifrost model ID must use the provider/model format: {model_id!r}."
        raise ValueError(msg)

    return {
        "model": upstream_model,
        "extra_headers": {"x-model-provider": provider},
    }


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
    desired_workspace_model_ids = {spec.workspace_model_id for spec in specs}
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
        workspace_model = _workspace_model_payload(spec)
        # Open WebUI preserves existing grants when imports omit this field.
        if spec.workspace_model_id not in existing_model_ids:
            workspace_model["access_grants"] = [PUBLIC_READ_GRANT]
        payloads.append(workspace_model)

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
    name: Any,
    schema: Any,
    default: Any,
) -> dict[str, Any] | None:
    if (
        not isinstance(name, str)
        or CHAT_VARIABLE_KEY.fullmatch(name) is None
        or not isinstance(schema, dict)
    ):
        return None

    label = str(schema.get("title") or name.replace("_", " ").title())
    schema_type = schema.get("type")
    if schema_type == "boolean" and type(default) is bool:
        return {
            "key": name,
            "type": "checkbox",
            "label": label,
            "default": default,
        }
    if schema_type != "string" or not isinstance(default, str):
        return None

    enum = schema.get("enum")
    if enum is None:
        return {
            "key": name,
            "type": "text",
            "label": label,
            "default": default,
        }
    if (
        not isinstance(enum, list)
        or not enum
        or any(not isinstance(value, str) for value in enum)
        or len(set(enum)) != len(enum)
        or default not in enum
    ):
        return None
    return {
        "key": name,
        "type": "select",
        "label": label,
        "options": enum,
        "default": default,
    }


def _model_extension(model: Any) -> dict[str, Any] | None:
    extension = (getattr(model, "model_extra", None) or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None
    description = extension.get("description")
    features = extension.get("features")
    if (
        not isinstance(features, list)
        or any(not isinstance(feature, str) for feature in features)
        or not isinstance(description, str)
        or not description.strip()
    ):
        return None
    return extension


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
    return {
        "id": spec.workspace_model_id,
        "base_model_id": spec.base_model_id,
        "name": spec.name,
        "meta": {
            "description": spec.description or LIMITED_FUNCTIONALITY_DESCRIPTION,
            CHAT_VARIABLES_META_KEY: {"fields": list(spec.fields)},
        },
        "params": {},
    }
