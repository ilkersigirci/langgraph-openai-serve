"""
Text, data, and code files, whose content is their own text.

No extractor reads them. A CSV or TSV file is text too: its text keeps every value exactly as written.
JSON stays out, since generated files, such as charts, are JSON that clients
display from their original bytes, and so does TypeScript, which browsers label
as video.
"""

import logging
from pathlib import PurePath

from lgos_files_api.api.files.content import text_content

logger = logging.getLogger(__name__)

SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".log",
    ".csv",
    ".tsv",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".py",
    ".js",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".css",
}


def is_supported(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in SUFFIXES


def extract(filename: str, document: bytes) -> bytes:
    """Return the text file's content as JSON. Blocking: run it in a thread."""
    try:
        text = document.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not UTF-8, using replacement characters", filename)
        text = document.decode("utf-8", errors="replace")
    return text_content(text)
