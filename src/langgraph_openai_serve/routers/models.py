from fastapi import APIRouter
from fastapi.responses import JSONResponse

from langgraph_openai_serve.models.openai_models import (
    Model,
    ModelList,
    ModelPermission,
)
from langgraph_openai_serve.services.graph_runner import GRAPH_REGISTRY

router = APIRouter(prefix="/models", tags=["openai"])


@router.get("/")
async def get_models():
    """Return a list of available models in OpenAI compatible format"""
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
            id=graph_name,
            created=1743771509,
            owned_by="langgraph-openai-serve",
            root=f"{graph_name}-root",
            parent=None,
            max_model_len=16000,
            permission=[permission],
        )
        for graph_name in GRAPH_REGISTRY
    ]

    model_list = ModelList(data=models)
    return JSONResponse(content=model_list.model_dump())
