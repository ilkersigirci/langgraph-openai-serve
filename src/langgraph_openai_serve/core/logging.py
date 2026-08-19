"""Request-scoped context for standard-library log records."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import TypeAlias

_LogContext: TypeAlias = dict[str, str | bool]

_log_context: ContextVar[_LogContext | None] = ContextVar("_log_context", default=None)
logging.getLogger("langgraph_openai_serve").addHandler(logging.NullHandler())


def begin_log_context(request_id: str) -> Token[_LogContext | None]:
    """Start a request context and return a token for restoring its parent."""
    return _log_context.set({"request_id": request_id})


def bind_log_context(
    *,
    model: str | None = None,
    stream: bool | None = None,
    run_id: str | None = None,
) -> None:
    """Add chat and graph fields without mutating the current context."""
    current = _log_context.get() or {}
    fields: _LogContext = {}
    if model is not None:
        fields["model"] = model
    if stream is not None:
        fields["stream"] = stream
    if run_id is not None:
        fields["run_id"] = run_id
    if fields:
        _log_context.set({**current, **fields})


def reset_log_context(token: Token[_LogContext | None]) -> None:
    """Restore the context that was active before a request started."""
    _log_context.reset(token)


class RequestContextFilter(logging.Filter):
    """Add active LGOS request fields to records emitted by LGOS loggers."""

    def __init__(self) -> None:
        super().__init__()
        self._log_context = _log_context

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich a record while preserving fields supplied by the caller."""
        for name, value in (self._log_context.get() or {}).items():
            if name not in record.__dict__:
                setattr(record, name, value)
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a normal logger with the LGOS context filter installed once."""
    logger = logging.getLogger(name)
    if not any(isinstance(item, RequestContextFilter) for item in logger.filters):
        logger.addFilter(RequestContextFilter())
    return logger
