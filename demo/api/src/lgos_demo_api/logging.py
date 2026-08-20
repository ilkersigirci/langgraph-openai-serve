"""JSON logging configuration owned by the demo application."""

import logging.config
from typing import Any

import structlog


class _DropUvicornColorMessage(logging.Filter):
    """Remove Uvicorn's redundant ANSI-formatted copy before any export."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.pop("color_message", None)
        return True


_FOREIGN_PRE_CHAIN = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.ExtraAdder(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]

_JSON_PROCESSORS = [
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.EventRenamer("message"),
    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    structlog.processors.JSONRenderer(),
]

LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "drop_uvicorn_color_message": {
            "()": _DropUvicornColorMessage,
        }
    },
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "foreign_pre_chain": _FOREIGN_PRE_CHAIN,
            "processors": _JSON_PROCESSORS,
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {
        "handlers": ["stdout"],
        "level": "WARNING",
    },
    "loggers": {
        "langgraph_openai_serve": {
            "level": "INFO",
            "propagate": True,
        },
        "lgos_demo_api": {
            "level": "INFO",
            "propagate": True,
        },
        "uvicorn": {
            # Let Uvicorn records reach the root handlers. This preserves the
            # stdout JSON stream and lets OTel's root LoggingHandler export the
            # same records when the OTel deployment overlay is enabled.
            "handlers": [],
            "level": "INFO",
            "propagate": True,
        },
        "uvicorn.error": {
            "handlers": [],
            "filters": ["drop_uvicorn_color_message"],
            "level": "INFO",
            "propagate": True,
        },
    },
}


def configure_logging() -> None:
    """Configure demo and server logs as JSON on the stdout stream."""
    logging.config.dictConfig(LOGGING_CONFIG)
