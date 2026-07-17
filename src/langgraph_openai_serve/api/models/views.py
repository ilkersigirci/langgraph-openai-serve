"""Models router.

This module provides the FastAPI router for the models endpoint,
implementing an OpenAI-compatible interface for model listing.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.models.schemas import ModelDetails, ModelList
from langgraph_openai_serve.api.models.service import ModelService
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.graph_registry import (
    GraphNotFoundError,
    GraphRegistry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["openai"])


def get_graph_registry_dependency(request: Request) -> GraphRegistry:
    """Dependency to get the graph registry from the app state."""
    return request.app.state.graph_registry


@router.get("", response_model=ModelList)
def list_models(
    service: Annotated[ModelService, Depends(ModelService)],
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
):
    """Get a list of available models."""
    logger.info("Received request to list models")
    models = service.get_models(graph_registry)
    logger.info(f"Returning {len(models.data)} models")
    return models


@router.get(
    "/{model}",
    response_model=ModelDetails,
    response_model_exclude_none=True,
)
def retrieve_model(
    model: str,
    service: Annotated[ModelService, Depends(ModelService)],
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
) -> ModelDetails:
    """Retrieve one registered graph as an OpenAI model."""
    logger.info(f"Received request to retrieve model: {model}")
    try:
        return service.get_model(model, graph_registry)
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
