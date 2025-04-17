"""Models router.

This module provides the FastAPI router for the models endpoint,
implementing an OpenAI-compatible interface for model listing.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from langgraph_openai_serve.api.models.service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["openai"])
model_service = ModelService()


@router.get("/")
async def get_models():
    """Return a list of available models in OpenAI compatible format"""
    model_list = model_service.get_models()
    return JSONResponse(content=model_list.model_dump())
