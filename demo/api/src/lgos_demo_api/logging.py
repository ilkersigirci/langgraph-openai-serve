"""JSON logging configuration owned by the demo application."""

import logging.config
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger


def _drop_uvicorn_color_message(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Remove Uvicorn's redundant ANSI-formatted copy of the message."""
    event_dict.pop("color_message", None)
    return event_dict


_FOREIGN_PRE_CHAIN = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.ExtraAdder(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]

_JSON_PROCESSORS = [
    _drop_uvicorn_color_message,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.EventRenamer("message"),
    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    structlog.processors.JSONRenderer(),
]

LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
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
            "stream": "ext://sys.stdout",
        }
    },
    "root": {
        "handlers": ["stdout"],
        "level": "INFO",
    },
    "loggers": {
        "uvicorn": {
            # Let Uvicorn records reach the root handlers. This preserves the
            # stdout JSON stream and lets OTel's root LoggingHandler export the
            # same records when the OTel deployment overlay is enabled.
            "handlers": [],
            "level": "INFO",
            "propagate": True,
        },
    },
}


def configure_logging() -> None:
    """Configure demo and server logs as JSON on the stdout stream."""
    logging.config.dictConfig(LOGGING_CONFIG)
