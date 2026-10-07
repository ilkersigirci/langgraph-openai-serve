"""Structured logging on stdout for the server and worker processes."""

import logging.config
import os
import sys
from typing import Any

import structlog
from structlog.typing import Processor

from langgraph_openai_serve.core.logging import RequestContextFilter


class _DropUvicornColorMessage(logging.Filter):
    """Remove Uvicorn's redundant ANSI-formatted copy before any export."""

    def filter(self, record: logging.LogRecord) -> bool:  # ruff: ignore[no-self-use] - Overrides logging.Filter.filter.
        record.__dict__.pop("color_message", None)
        return True


def logging_config(application: str, *, root_level: int) -> dict[str, Any]:
    """Return a ``dictConfig`` that logs INFO from LGOS, Uvicorn, and the app."""
    info = {"level": "INFO", "propagate": True}
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "drop_uvicorn_color_message": {"()": _DropUvicornColorMessage},
            "request_context": {"()": RequestContextFilter},
        },
        "formatters": {
            "structlog": {
                "()": structlog.stdlib.ProcessorFormatter,
                "foreign_pre_chain": [
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.ExtraAdder(),
                    structlog.processors.TimeStamper(fmt="iso", utc=True),
                ],
                "processors": [
                    structlog.processors.StackInfoRenderer(),
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    *_renderers(),
                ],
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "structlog",
                # Records from graph nodes and dependencies also get request IDs.
                "filters": ["request_context"],
                "level": "INFO",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"handlers": ["stdout"], "level": root_level},
        "loggers": {
            application: {**info},
            "hatchet": {**info, "handlers": []},
            "langgraph_openai_serve": {**info},
            # Uvicorn records reach the root handler, so OpenTelemetry's root
            # LoggingHandler exports them too when the launcher is used.
            "uvicorn": {**info, "handlers": []},
            "uvicorn.error": {
                **info,
                "handlers": [],
                "filters": ["drop_uvicorn_color_message"],
            },
        },
    }


def _renderers() -> list[Processor]:
    """Render readable lines for a person at a terminal and JSON for collectors."""
    if sys.stdout.isatty():
        # ConsoleRenderer formats exceptions itself. Like structlog's default
        # configuration, honor the NO_COLOR convention.
        return [structlog.dev.ConsoleRenderer(colors=not os.environ.get("NO_COLOR"))]
    return [
        structlog.processors.format_exc_info,
        structlog.processors.EventRenamer("message"),
        structlog.processors.JSONRenderer(),
    ]


def configure_logging(application: str, *, root_level: int) -> None:
    """Install the logging configuration for a process that Uvicorn does not run."""
    logging.config.dictConfig(logging_config(application, root_level=root_level))
