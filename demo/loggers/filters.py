"""Filters for logging."""

import logging
import threading
from pathlib import Path

from demo.utils import is_module_installed

PATHNAME_FIELD = "pathname"
PATHNAME_MAX_LENGTH = 30
NEST_COUNT = 2


class InfoFilter(logging.Filter):
    """This filter only shows log entries for INFO level."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter function."""
        return record.levelno == logging.INFO


class SimpleThreadFilter(logging.Filter):
    """This filter only shows log entries for specified thread name.

    Args:
        thread_name: Name of the thread that will be filtered.
    """

    def __init__(self, thread_name: str):
        self.thread_name = thread_name
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter function."""
        return record.threadName == self.thread_name


class ThreadFilter(logging.Filter):
    """Only accept log records from a specific thread or thread name.

    Args:
        thread_id: Id of the thread that will be filtered.
        thread_name: Name of the thread that will be filtered.

    Raises:
        ValueError: Occurs when `thread_id` and/or `thread_id` not given.
    """

    def __init__(
        self, thread_id: int | None = None, thread_name: str | None = None
    ) -> None:
        if thread_id is None and thread_name is None:
            raise ValueError("Must specify either thread_id or thread_name")

        self._thread_id = thread_id
        self._thread_name = thread_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter function."""
        if self._thread_id is not None and record.thread != self._thread_id:
            return False

        return not (
            self._thread_name is not None and record.threadName != self._thread_name
        )


class IgnoreThreadsFilter(logging.Filter):
    """Only accepts log records that originated from the main thread.

    Attributes:
        _main_thread_id: Id of the main thread.
    """

    def __init__(self) -> None:
        self._main_thread_id = threading.main_thread().ident

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter function."""
        return record.thread == self._main_thread_id


class RemoveColorFilter(logging.Filter):
    """Remove color filter."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter function."""
        is_module_installed(module_name="click", throw_error=True)
        import click  # noqa: PLC0415

        if record and record.msg and isinstance(record.msg, str):
            record.msg = click.unstyle(record.msg)

        return True


class CwdFilter(logging.Filter):
    """Filter that removes cwd from pathname."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter function."""
        record.pathname = record.pathname.replace(str(Path.cwd()), "")

        return True
