from fastapi import APIRouter
from fastapi.responses import JSONResponse

from langgraph_openai_serve.models.openai_models import (
    Model,
    ModelList,
    ModelPermission,
)

router = APIRouter(prefix="/v1", tags=["openai"])


@router.get("/models")
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

    model = Model(
        id="test-model",
        created=1743771509,
        owned_by="databoss",
        root="test-model-root-path",
        parent=None,
        max_model_len=16000,
        permission=[permission],
    )

    model_list = ModelList(data=[model])
    return JSONResponse(content=model_list.model_dump())
