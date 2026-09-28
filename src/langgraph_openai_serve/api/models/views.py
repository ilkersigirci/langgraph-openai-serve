"""OpenAI-compatible model listing and retrieval."""

from typing import Annotated

from fastapi import APIRouter, Depends

from langgraph_openai_serve.api.deps import get_graph_registry
from langgraph_openai_serve.api.models import service as models_service
from langgraph_openai_serve.api.models.schemas import ModelDetails, ModelList
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

router = APIRouter(prefix="/models", tags=["openai"])


@router.get("")
def list_models(
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry)],
) -> ModelList:
    """Get a list of available models."""
    return models_service.get_models(graph_registry)


@router.get("/{model}", response_model_exclude_none=True)
def retrieve_model(
    model: str,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry)],
) -> ModelDetails:
    """Retrieve one registered graph as an OpenAI model."""
    return models_service.get_model(model, graph_registry)
