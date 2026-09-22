"""OpenAI-compatible Responses router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, status
from fastapi.responses import StreamingResponse
from openai.types.responses import Response
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.deps import get_graph_registry, get_stream_owner
from langgraph_openai_serve.api.errors import graph_errors
from langgraph_openai_serve.api.responses import service as responses_service
from langgraph_openai_serve.api.responses.deps import (
    get_background_backend,
    get_checkpoint_scope,
    validate_background_retrieval,
)
from langgraph_openai_serve.api.responses.messages import InvalidResponsesInputError
from langgraph_openai_serve.api.responses.output import UnsupportedResponsesOutputError
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.streaming import StreamOwner
from langgraph_openai_serve.background.contracts import BackgroundBackend
from langgraph_openai_serve.background.store import (
    BackgroundCapacityError,
    BackgroundIdempotencyConflictError,
    BackgroundResponseExpiredError,
)
from langgraph_openai_serve.core.errors import OpenAIHTTPException
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
    """Create one stateless OpenAI Response, optionally as an SSE stream."""
    bind_log_context(model=response_request.model, stream=response_request.stream)

    with graph_errors(input_param="input"):
        if response_request.background:
            try:
                return await responses_service.accept_background_response(
                    response_request,
                    graph_registry,
                    background,
                    checkpoint_scope=checkpoint_scope,
                    idempotency_key=idempotency_key,
                )
            except (
                UnsupportedResponsesRequestError,
                InvalidResponsesInputError,
            ) as exc:
                raise _invalid_request(exc) from exc
            except BackgroundIdempotencyConflictError as exc:
                message = (
                    "The background create idempotency key was already used with "
                    "different request content."
                )
                raise _idempotency_conflict(
                    message,
                    code="background_idempotency_conflict",
                ) from exc
            except BackgroundResponseExpiredError as exc:
                message = "The idempotent background Response has expired."
                raise _idempotency_conflict(
                    message,
                    code="background_response_expired",
                ) from exc
            except BackgroundCapacityError as exc:
                raise OpenAIHTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    error=ErrorObject(
                        message="Background response admission capacity is full.",
                        type="server_error",
                        code="background_capacity_exceeded",
                    ),
                ) from exc
        try:
            run = await responses_service.prepare_response_run(
                response_request,
                graph_registry,
                checkpoint_scope=checkpoint_scope,
            )
        except (UnsupportedResponsesRequestError, InvalidResponsesInputError) as exc:
            raise _invalid_request(exc) from exc
        if response_request.stream:
            body = stream_owner.start(
                responses_service.stream_response(response_request, run),
                run,
            )
            return StreamingResponse(body, media_type="text/event-stream")
        try:
            return await responses_service.collect_response(response_request, run)
        except UnsupportedResponsesOutputError as exc:
            raise OpenAIHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error=ErrorObject(message=str(exc), type="server_error"),
            ) from exc


@router.get("/responses/{response_id}")
async def retrieve_response(
    response_id: str,
    checkpoint_scope: Annotated[str, Depends(get_checkpoint_scope)],
    background: Annotated[
        BackgroundBackend | None,
        Depends(get_background_backend),
    ],
    _retrieval_validation: Annotated[
        None,
        Depends(validate_background_retrieval),
    ],
) -> Response:
    """Retrieve one authorized background Response snapshot."""
    with graph_errors(input_param="input"):
        try:
            return await responses_service.retrieve_background_response(
                response_id,
                background,
                checkpoint_scope=checkpoint_scope,
            )
        except responses_service.BackgroundResponseNotFoundError as exc:
            raise _not_found(response_id) from exc


@router.post("/responses/{response_id}/cancel")
async def cancel_response(
    response_id: str,
    checkpoint_scope: Annotated[str, Depends(get_checkpoint_scope)],
    background: Annotated[
        BackgroundBackend | None,
        Depends(get_background_backend),
    ],
) -> Response:
    """Atomically cancel one authorized active background Response."""
    with graph_errors(input_param="input"):
        try:
            return await responses_service.cancel_background_response(
                response_id,
                background,
                checkpoint_scope=checkpoint_scope,
            )
        except responses_service.BackgroundResponseNotFoundError as exc:
            raise _not_found(response_id) from exc


def _invalid_request(
    exc: UnsupportedResponsesRequestError | InvalidResponsesInputError,
) -> OpenAIHTTPException:
    return OpenAIHTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        error=ErrorObject(
            message=str(exc),
            type="invalid_request_error",
            param=(
                exc.param
                if isinstance(exc, UnsupportedResponsesRequestError)
                else "input"
            ),
        ),
    )


def _idempotency_conflict(
    message: str,
    *,
    code: str,
) -> OpenAIHTTPException:
    return OpenAIHTTPException(
        status_code=status.HTTP_409_CONFLICT,
        error=ErrorObject(
            message=message,
            type="invalid_request_error",
            param="Idempotency-Key",
            code=code,
        ),
    )


def _not_found(response_id: str) -> OpenAIHTTPException:
    message = f"Response '{response_id}' was not found."
    return OpenAIHTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        error=ErrorObject(
            message=message,
            type="invalid_request_error",
            param="response_id",
            code="response_not_found",
        ),
    )


__all__ = ["router"]
