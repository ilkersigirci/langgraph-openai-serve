"""Model service.

This module provides a service for handling OpenAI model information.
"""

import logging

from langgraph_openai_serve.graph.runner import get_graph_registry
from langgraph_openai_serve.schemas.openai_schema import (
    Model,
    ModelList,
    ModelPermission,
)

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model operations."""

    def get_models(self) -> ModelList:
        """Get a list of available models.

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

        graph_registry = get_graph_registry()

        models = [
            Model(
                id=graph_name,
                created=1743771509,
                owned_by="langgraph-openai-serve",
                root=f"{graph_name}-root",
                parent=None,
                max_model_len=16000,
                permission=[permission],
            )
            for graph_name in graph_registry
        ]

        logger.info(f"Retrieved {len(models)} available models")
        return ModelList(data=models)
