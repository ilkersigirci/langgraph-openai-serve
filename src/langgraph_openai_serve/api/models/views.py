"""Models router.

This module provides the FastAPI router for the models endpoint,
implementing an OpenAI-compatible interface for model listing.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from langgraph_openai_serve.api.models.schemas import ModelList
from langgraph_openai_serve.api.models.service import ModelService
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

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
