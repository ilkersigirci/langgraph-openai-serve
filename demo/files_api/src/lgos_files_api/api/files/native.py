"""
Text files and spreadsheets, extracted here instead of by xberg.

The content is a JSON object with a `type` and its `version`:

- text (.md, .txt): {"type": "text", "version": 1, "text": "..."}; xberg's
  documents share it, with their Markdown as the text.
- spreadsheet (.xlsx, .xls): {"type": "spreadsheet", "version": 1, "sheets":
  [{"name", "headers", "rows", "chunks"}]}: per sheet the header row, the data
  rows padded to the header width, and the rows rendered as markdown chunks.
"""

import csv
import io
import json
import logging
import tempfile
from collections.abc import Mapping
from itertools import starmap
from pathlib import PurePath
from typing import TypedDict

import xlrd
from xlsx2csv import Xlsx2csv

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".txt"}
SPREADSHEET_SUFFIXES = {".xlsx", ".xls"}
NATIVE_SUFFIXES = TEXT_SUFFIXES | SPREADSHEET_SUFFIXES
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


def is_supported(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in NATIVE_SUFFIXES


def text_content(text: str) -> bytes:
    """Return the stored JSON of a text content."""
    return _json({"type": "text", "version": TEXT_VERSION, "text": text})


def extract(filename: str, document: bytes) -> bytes:
    """
    Return the document's content as JSON. Blocking: run it in a thread.

    Raises ValueError for documents that can't be read.
    """
    suffix = PurePath(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return text_content(_decode(document, filename))
    if suffix not in SPREADSHEET_SUFFIXES:
        msg = f"No native extractor for {filename!r}"
        raise ValueError(msg)
    try:
        sheets = _xlsx_sheets(document) if suffix == ".xlsx" else _xls_sheets(document)
    # Each reader raises its own errors on unreadable files.
    except Exception as error:
        msg = f"Unreadable spreadsheet: {error!r}"
        raise ValueError(msg) from error
    return _json(
        {
            "type": "spreadsheet",
            "version": SPREADSHEET_VERSION,
            "sheets": list(starmap(_sheet, sheets)),
        }
    )


def _json(content: Mapping[str, object]) -> bytes:
    return json.dumps(content, ensure_ascii=False).encode()


def _decode(document: bytes, filename: str) -> str:
    try:
        return document.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not UTF-8, using replacement characters", filename)
        return document.decode("utf-8", errors="replace")


def _xlsx_sheets(document: bytes) -> list[tuple[str, str]]:
    """Return (sheet name, CSV) per sheet."""
    # xlsx2csv reads only paths.
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as temp_file:
        temp_file.write(document)
        temp_file.flush()
        workbook = Xlsx2csv(
            temp_file.name,
            outputencoding="utf-8",
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
            sheetdelimiter="",
            skip_empty_lines=False,
        )
        try:
            sheets = []
            for index, sheet in enumerate(getattr(workbook.workbook, "sheets", [])):
                csv_buffer = io.StringIO()
                workbook.convert(
                    csv_buffer,
                    sheetid=sheet.get("index") or sheet.get("id") or index + 1,
                )
                sheets.append(
                    (sheet.get("name") or f"Sheet {index + 1}", csv_buffer.getvalue())
                )
            return sheets
        finally:
            if (ziphandle := getattr(workbook, "ziphandle", None)) is not None:
                ziphandle.close()


def _xls_sheets(document: bytes) -> list[tuple[str, str]]:
    """Return (sheet name, CSV) per sheet."""
    workbook = xlrd.open_workbook(file_contents=document)
    sheets = []
    for index in range(workbook.nsheets):
        sheet = workbook.sheet_by_index(index)
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
        for row_index in range(sheet.nrows):
            writer.writerow(
                [
                    str(cell) if cell is not None else ""
                    for cell in sheet.row_values(row_index)
                ]
            )
        sheets.append((sheet.name or f"Sheet {index + 1}", csv_buffer.getvalue()))
    return sheets


def _sheet(name: str, csv_content: str) -> Sheet:
    """Return the sheet's header row, data rows, and markdown chunks of row groups."""
    rows = [_trim_row(row) for row in csv.reader(io.StringIO(csv_content))]
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
    normalized = str(cell or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("\n", "<br>").replace("|", r"\|")


def _trim_row(row: list[str]) -> list[str]:
    trimmed = [str(cell or "").strip() for cell in row]
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
