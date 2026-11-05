"""OpenAI-compatible Responses router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import StreamingResponse
from openai.types.responses import Response

from langgraph_openai_serve.api.deps import get_graph_registry, get_stream_owner
from langgraph_openai_serve.api.responses import (
    background as responses_background,
    service as responses_service,
)
from langgraph_openai_serve.api.responses.deps import (
    get_background_backend,
    get_checkpoint_scope,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.streaming import StreamOwner
from langgraph_openai_serve.background import BackgroundBackend
from langgraph_openai_serve.core.logging import bind_log_context
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

router = APIRouter(tags=["openai"])


@router.post("/responses", response_model=Response)
async def create_response(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - Explicit FastAPI dependencies.
    response_request: ResponseCreateRequest,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry)],
    checkpoint_scope: Annotated[str, Depends(get_checkpoint_scope)],
    background: Annotated[
        BackgroundBackend | None,
        Depends(get_background_backend),
    ],
    stream_owner: Annotated[
        StreamOwner,
        Depends(get_stream_owner, scope="request"),
    ],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
) -> StreamingResponse | Response:
    """Create one OpenAI Response, as JSON, an SSE stream, or a background run."""
    bind_log_context(model=response_request.model, stream=response_request.stream)
    if response_request.background:
        accepted = await responses_background.create_background_response(
            response_request,
            graph_registry,
            background,
            checkpoint_scope=checkpoint_scope,
            idempotency_key=idempotency_key,
        )
        if isinstance(accepted, Response):
            return accepted
        return stream_owner.follow(accepted)
    run = await responses_service.prepare_response_run(
        response_request,
        graph_registry,
        checkpoint_scope=checkpoint_scope,
    )
    if response_request.stream:
        return stream_owner.start(
            responses_service.stream_response(response_request, run),
            run,
        )
    return await responses_service.collect_response(response_request, run)


@router.get("/responses/{response_id}", response_model=Response)
async def retrieve_response(  # ruff: ignore[too-many-arguments] - Explicit FastAPI dependencies.
    response_id: str,
    checkpoint_scope: Annotated[str, Depends(get_checkpoint_scope)],
    background: Annotated[
        BackgroundBackend | None,
        Depends(get_background_backend),
    ],
    stream_owner: Annotated[
        StreamOwner,
        Depends(get_stream_owner, scope="request"),
    ],
    *,
    stream: bool = False,
    starting_after: Annotated[int | None, Query(ge=0)] = None,
) -> StreamingResponse | Response:
    """Retrieve one authorized background Response as JSON or an SSE stream."""
    retrieved = await responses_background.retrieve_background_response(
        response_id,
        background,
        checkpoint_scope=checkpoint_scope,
        stream=stream,
        starting_after=starting_after,
    )
    if isinstance(retrieved, Response):
        return retrieved
    return stream_owner.follow(retrieved)


@router.post("/responses/{response_id}/cancel")
async def cancel_response(
    response_id: str,
    checkpoint_scope: Annotated[str, Depends(get_checkpoint_scope)],
    background: Annotated[
        BackgroundBackend | None,
        Depends(get_background_backend),
    ],
) -> Response:
    """Cancel one authorized active background Response."""
    return await responses_background.cancel_background_response(
        response_id,
        background,
        checkpoint_scope=checkpoint_scope,
    )
