"""OpenAI-compatible Chat Completions router."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion

from langgraph_openai_serve.api.chat import service as chat_service
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.api.deps import get_graph_registry, get_stream_owner
from langgraph_openai_serve.api.streaming import StreamOwner
from langgraph_openai_serve.core.logging import bind_log_context
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

router = APIRouter(tags=["openai"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletion,
    response_model_exclude_none=True,
)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry)],
    stream_owner: Annotated[
        StreamOwner,
        Depends(get_stream_owner, scope="request"),
    ],
) -> StreamingResponse | ChatCompletion:
    """Create one chat completion, as JSON or as an SSE stream."""
    bind_log_context(model=chat_request.model, stream=chat_request.stream)
    run = await chat_service.prepare_completion_run(chat_request, graph_registry)
    if chat_request.stream:
        return stream_owner.start(
            chat_service.stream_completion(chat_request, run),
            run,
        )
    return await chat_service.generate_completion(chat_request, run)
