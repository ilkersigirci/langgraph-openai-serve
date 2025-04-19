from pydantic import BaseModel


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


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[Model]
