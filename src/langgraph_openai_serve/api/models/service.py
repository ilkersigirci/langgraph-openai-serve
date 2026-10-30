"""Functions for building OpenAI model information."""

from itertools import starmap

from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    Model,
    ModelClientSettings,
    ModelList,
)
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry

MODEL_CREATED = 1743771509
MODEL_OWNER = "langgraph-openai-serve"


def get_models(graph_registry: GraphRegistry) -> ModelList:
    """List every registered graph as an OpenAI model."""
    return ModelList(data=list(starmap(_model, graph_registry.graphs.items())))


def get_model(model: str, graph_registry: GraphRegistry) -> Model:
    """Get one registered graph as an OpenAI model with its client settings."""
    graph_config = graph_registry.get_graph(model)
    settings = graph_config.client_settings
    return _model(
        model,
        graph_config,
        client_settings=(
            ModelClientSettings(
                json_schema=settings.json_schema(),
                defaults=settings.default_values(),
            )
            if settings is not None
            else None
        ),
    )


def _model(
    name: str,
    graph_config: GraphConfig,
    client_settings: ModelClientSettings | None = None,
) -> Model:
    return Model(
        id=name,
        created=MODEL_CREATED,
        owned_by=MODEL_OWNER,
        lgos=LangGraphModelExtension(
            description=graph_config.description,
            features=sorted(graph_config.features),
            server_tools=sorted(graph_config.server_tools),
            client_tools=sorted(graph_config.client_tools),
            client_settings=client_settings,
        ),
    )
