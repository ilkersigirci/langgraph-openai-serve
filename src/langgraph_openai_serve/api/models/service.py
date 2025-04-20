"""Model service.

This module provides a service for handling OpenAI model information.
"""

import logging

from langgraph_openai_serve.api.models.schemas import (
    Model,
    ModelList,
    ModelPermission,
)
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model operations."""

    def get_models(self, graph_registry: GraphRegistry) -> ModelList:
        """Get a list of available models.

        Args:
            graph_registry: The GraphRegistry containing registered graphs.

        Returns:
            A list of models in OpenAI compatible format.
        """
        permission = ModelPermission(
            id="modelperm-04cadfeee8ad4eb8ad479a5af3bc261d",
            created=1743771509,
            allow_create_engine=False,
            allow_sampling=True,
            allow_logprobs=True,
            allow_search_indices=False,
            allow_view=True,
            allow_fine_tuning=False,
            organization="*",
            group=None,
            is_blocking=False,
        )

        models = [
            Model(
                id=name,
                created=1743771509,
                owned_by="langgraph-openai-serve",
                root=f"{name}-root",
                parent=None,
                max_model_len=16000,
                permission=[permission],
            )
            for name in graph_registry.registry
        ]

        logger.info(f"Retrieved {len(models)} available models")
        return ModelList(data=models)
