"""The model catalog that ``lgos serve`` and ``lgos worker`` run."""

from langchain_core.language_models import BaseChatModel
from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRegistry
from langgraph_openai_serve.server import ServerResources

from {{ cookiecutter.project_slug }}.graphs.approval import (
    create_approval_graph,
)
from {{ cookiecutter.project_slug }}.graphs.simple import (
    SimpleContext,
    create_simple_graph,
)


def create_registry(
    resources: ServerResources,
    *,
    model: BaseChatModel | None = None,
) -> GraphRegistry:
    return GraphRegistry(
        graphs={
            "simple-graph": GraphConfig(
                graph=create_simple_graph(model),
                description="Answer questions with configurable history and audience.",
                client_settings=SimpleContext,
                features={GraphFeature.BACKGROUND},
            ),
            "approval": GraphConfig(
                graph=create_approval_graph(resources.checkpointer),
                description="Pause for human approval before completing a request.",
                features={GraphFeature.INTERRUPTS, GraphFeature.BACKGROUND},
            ),
        },
        run_coordinator=resources.run_coordinator,
    )
