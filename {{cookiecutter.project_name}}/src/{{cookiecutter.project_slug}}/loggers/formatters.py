"""Formatters for the logger."""

import logging
from enum import Enum


class _TerminalLogColor(str, Enum):
    DEBUG = "94m"
    INFO = "92m"
    WARNING = "93m"
    ERROR = "91m"
    CRITICAL = "95m"


class ColoredFormatter(logging.Formatter):
    """Formatter for colored logs.

    It uses ansi escape codes to color the logs
    in the terminal.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        log_level = record.levelname

        log_color = _TerminalLogColor[log_level].value
        formatted_record = super().format(record)

        return f"\033[{log_color}{formatted_record}\033[0m"
