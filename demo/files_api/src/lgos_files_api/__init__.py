"""Standalone OpenAI-compatible Files API."""

from lgos_files_api.app import create_files_app
from lgos_files_api.contracts import FileRepository

__all__ = ["FileRepository", "create_files_app"]
