"""Translate shared graph failures at either OpenAI inference boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from fastapi import status
from openai.types.shared import ErrorObject

from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.client_settings import ClientSettingsValidationError
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfigurationError,
    GraphNotFoundError,
)
from langgraph_openai_serve.graph.interrupt.coordination import RunBusyError
from langgraph_openai_serve.graph.interrupt.errors import (
    InvalidInterruptPayloadError,
    InvalidResumeRequestError,
)
from langgraph_openai_serve.graph.interrupt.state import (
    RUN_METADATA_KEY,
    InterruptStateConflictError,
    InvalidRunIDError,
)


@contextmanager
def graph_errors(*, input_param: Literal["input", "messages"]) -> Iterator[None]:
    """
    Map graph errors to OpenAI errors using the endpoint's input field.

    Yields:
        Control to request decoding, preparation, and non-streaming execution.

    """
    try:
        yield
    except (RunBusyError, InterruptStateConflictError) as exc:
        busy = isinstance(exc, RunBusyError)
        raise OpenAIHTTPException(
            status_code=status.HTTP_409_CONFLICT,
            error=ErrorObject(
                message=str(exc),
                type="invalid_request_error",
                param=None if busy else input_param,
                code="run_busy" if busy else "interrupt_state_conflict",
            ),
        ) from exc
    except (
        InvalidRunIDError,
        InvalidResumeRequestError,
        GraphNotFoundError,
        ClientSettingsValidationError,
    ) as exc:
        match exc:
            case InvalidRunIDError():
                param = f"metadata.{RUN_METADATA_KEY}"
            case GraphNotFoundError():
                param = "model"
            case ClientSettingsValidationError():
                param = exc.param
            case InvalidResumeRequestError():
                param = input_param
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message=str(exc), type="invalid_request_error", param=param
            ),
        ) from exc
    except (GraphConfigurationError, InvalidInterruptPayloadError) as exc:
        raise OpenAIHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error=ErrorObject(message=str(exc), type="server_error"),
        ) from exc
