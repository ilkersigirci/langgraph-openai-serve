"""
Models router.

This module provides the FastAPI router for the models endpoint,
implementing an OpenAI-compatible interface for model listing.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.models import service as models_service
from langgraph_openai_serve.api.models.deps import get_graph_registry_dependency
from langgraph_openai_serve.api.models.schemas import ModelDetails, ModelList
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.graph_registry import (
    GraphNotFoundError,
    GraphRegistry,
)

router = APIRouter(prefix="/models", tags=["openai"])


@router.get("")
def list_models(
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
) -> ModelList:
    """Get a list of available models."""
    return models_service.get_models(graph_registry)


@router.get(
    "/{model}",
    response_model_exclude_none=True,
)
def retrieve_model(
    model: str,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
) -> ModelDetails:
    """Retrieve one registered graph as an OpenAI model."""
    try:
        return models_service.get_model(model, graph_registry)
    except GraphNotFoundError as exc:
        raise OpenAIHTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            error=ErrorObject(
                message=str(exc),
                type="invalid_request_error",
                param="model",
                code="model_not_found",
            ),
        ) from exc
