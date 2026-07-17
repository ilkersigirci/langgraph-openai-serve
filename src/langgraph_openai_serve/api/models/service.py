"""Model service.

This module provides a service for handling OpenAI model information.
"""

import logging

from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    Model,
    ModelClientSettings,
    ModelDetails,
    ModelList,
)
from langgraph_openai_serve.graph.client_settings import (
    client_settings_default_values,
    client_settings_json_schema,
)
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = logging.getLogger(__name__)
MODEL_CREATED = 1743771509
MODEL_OWNER = "langgraph-openai-serve"


class ModelService:
    """Service for handling model operations."""

    def get_models(self, graph_registry: GraphRegistry) -> ModelList:
        """Get a list of available models.

        Args:
            graph_registry: The GraphRegistry containing registered graphs.

        Returns:
            A list of models in OpenAI compatible format.
        """
        models = [
            Model(
                id=name,
                created=MODEL_CREATED,
                owned_by=MODEL_OWNER,
            )
            for name in graph_registry.registry
        ]

        logger.info(f"Retrieved {len(models)} available models")
        return ModelList(data=models)

    def get_model(self, model: str, graph_registry: GraphRegistry) -> ModelDetails:
        """Get one registered graph as an OpenAI model with LGOS metadata."""
        graph_config = graph_registry.get_graph(model)
        client_settings = graph_config.client_settings
        client_settings_details = None
        if client_settings is not None:
            client_settings_details = ModelClientSettings(
                json_schema=client_settings_json_schema(client_settings),
                defaults=client_settings_default_values(client_settings),
            )

        details = ModelDetails(
            id=model,
            created=MODEL_CREATED,
            owned_by=MODEL_OWNER,
            langgraph_openai_serve=LangGraphModelExtension(
                features=sorted(
                    graph_config.features,
                    key=lambda feature: feature.value,
                ),
                client_settings=client_settings_details,
            ),
        )
        logger.info(f"Retrieved model details for {model}")
        return details
