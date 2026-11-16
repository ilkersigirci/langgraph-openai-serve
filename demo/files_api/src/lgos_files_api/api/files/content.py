"""
The JSON content stored for an extracted file, as clients read it.

Each content has a `type` and its `version`:

- text: {"type": "text", "version": 1, "text": "..."}: a text file's own text,
  or a document's Markdown.
- spreadsheet: {"type": "spreadsheet", "version": 1, "sheets": [{"name",
  "headers", "rows", "chunks"}]}: per sheet the header row, the data rows padded
  to the header width, and the rows rendered as markdown chunks.
"""

import json
from collections.abc import Iterable, Mapping
from itertools import starmap
from typing import TypedDict

TEXT_VERSION = 1
SPREADSHEET_VERSION = 1
# Chunks of consecutive rows stay within both limits; a single row may exceed
# the character limit.
MAX_CHUNK_CHARACTERS = 4000
MAX_ROWS_PER_CHUNK = 50

Row = tuple[int, list[str]]


class Sheet(TypedDict):
    name: str
    headers: list[str]
    rows: list[list[str]]
    chunks: list[str]


def text_content(text: str) -> bytes:
    """Return the stored JSON of a text content."""
    return _json({"type": "text", "version": TEXT_VERSION, "text": text})


def spreadsheet_content(sheets: Iterable[tuple[str, list[list[str]]]]) -> bytes:
    """Return the stored JSON of a spreadsheet, from each sheet's name and cells."""
    return _json(
        {
            "type": "spreadsheet",
            "version": SPREADSHEET_VERSION,
            "sheets": list(starmap(_sheet, sheets)),
        }
    )


def _json(content: Mapping[str, object]) -> bytes:
    return json.dumps(content, ensure_ascii=False).encode()


def _sheet(name: str, cells: list[list[str]]) -> Sheet:
    """Return the sheet's header row, data rows, and markdown chunks of row groups."""
    rows = [_trim_row(row) for row in cells]
    non_empty_rows = [row for row in rows if any(row)]
    if not non_empty_rows:
        return {"name": name, "headers": [], "rows": [], "chunks": []}

    headers, data_rows = non_empty_rows[0], non_empty_rows[1:]
    column_count = max(len(headers), max((len(row) for row in data_rows), default=0))
    headers += [""] * (column_count - len(headers))
    numbered_rows = [
        (row_number, row + [""] * (column_count - len(row)))
        for row_number, row in enumerate(data_rows, start=1)
    ]
    return {
        "name": name,
        "headers": headers,
        "rows": [row for _, row in numbered_rows],
        "chunks": [
            _rows_text(name, headers, group)
            for group in _row_groups(name, headers, numbered_rows)
        ],
    }


def _normalize_cell(cell: str) -> str:
    normalized = cell.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("\n", "<br>").replace("|", r"\|")


def _trim_row(row: list[str]) -> list[str]:
    trimmed = [cell.strip() for cell in row]
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    return trimmed


def _rows_text(sheet_name: str, headers: list[str], rows: list[Row]) -> str:
    """Render 'Sheet: …', 'Fields: …', and a 'Row N | Header: value | …' line per row."""
    normalized_headers = [
        _normalize_cell(header) or f"Column {index + 1}"
        for index, header in enumerate(headers)
    ]
    lines = [
        f"Sheet: {_normalize_cell(sheet_name)}",
        f"Fields: {', '.join(normalized_headers)}",
        "",
    ]
    for row_number, row in rows:
        fields = [
            f"{header}: {cell}"
            for header, cell in zip(
                normalized_headers, map(_normalize_cell, row), strict=False
            )
            if cell
        ]
        if fields:
            lines.append(f"Row {row_number} | " + " | ".join(fields))
    return "\n".join(lines)


def _row_groups(
    sheet_name: str, headers: list[str], rows: list[Row]
) -> list[list[Row]]:
    """Group consecutive rows into chunks within the row and character limits."""
    if not headers:
        return []
    if not rows:
        return [[]]
    max_characters = max(MAX_CHUNK_CHARACTERS, len(_rows_text(sheet_name, headers, [])))

    groups: list[list[Row]] = []
    current: list[Row] = []
    for row in rows:
        candidate = [*current, row]
        if current and (
            len(current) >= MAX_ROWS_PER_CHUNK
            or len(_rows_text(sheet_name, headers, candidate)) > max_characters
        ):
            groups.append(current)
            current = [row]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups
