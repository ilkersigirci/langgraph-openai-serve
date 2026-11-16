"""
Documents and spreadsheets, extracted with xberg in this process.

PDF, Office, OpenDocument, RTF, and HTML documents: xberg reads a document's
text layer and OCRs pages without one; its content is a text content holding
the document's Markdown. A spreadsheet's content holds each sheet's cells as
xberg formats them, such as dates as ISO dates.
"""

from pathlib import PurePath

import xberg

from lgos_files_api.api.files.content import spreadsheet_content, text_content

DOCUMENT_SUFFIXES = {
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".html",
    ".htm",
    ".pptx",
    ".ppt",
}
SPREADSHEET_SUFFIXES = {".xlsx", ".xls"}
SUFFIXES = DOCUMENT_SUFFIXES | SPREADSHEET_SUFFIXES
# The container image bundles English OCR data only; xberg would download the
# data of other languages on first use.
OCR_LANGUAGES = ["eng"]


async def extract(
    filename: str, document: bytes, *, timeout_seconds: int, threads: int
) -> bytes:
    """
    Return the document's content as JSON, extracted with at most `threads` threads.

    Raises TimeoutError for documents xberg doesn't read within `timeout_seconds`,
    and ValueError for those it can't read.
    """
    config = xberg.ExtractionConfig(
        output_format="markdown",
        extraction_timeout_secs=timeout_seconds,
        concurrency=xberg.ConcurrencyConfig(
            max_threads=threads, max_concurrent_ocr=threads
        ),
        # S3 stores each result once; xberg's caches would write to the
        # container's read-only home directory.
        use_cache=False,
        ocr=xberg.OcrConfig(
            language=OCR_LANGUAGES,
            tesseract_config=xberg.TesseractConfig(
                language=OCR_LANGUAGES, use_cache=False
            ),
        ),
    )
    try:
        result = await xberg.extract(
            xberg.ExtractInput(kind="bytes", bytes=document, filename=filename),  # ty: ignore[invalid-argument-type] - xberg annotates the field with its own name, which ty reads as None.
            config,
        )
    # Whatever xberg raises means it can't read this document: xberg raises
    # RuntimeError, not its XbergError subclasses, for unreadable documents and
    # timeouts alike, and its async bridge reports a Rust panic as its own
    # exception.
    except Exception as error:
        if isinstance(error, xberg.XbergTimeoutError) or str(error).startswith(
            "Extraction timed out"
        ):
            raise TimeoutError(str(error)) from error
        raise ValueError(str(error)) from error
    if result.errors:
        raise ValueError(result.errors[0].message)
    if PurePath(filename).suffix.lower() in SPREADSHEET_SUFFIXES:
        (workbook,) = result.results
        return spreadsheet_content(_sheets(workbook))
    return text_content("\n\n".join(extracted.content for extracted in result.results))


def _sheets(workbook: xberg.ExtractedDocument) -> list[tuple[str, list[list[str]]]]:
    """Return each sheet's name and cells, in the workbook's order."""
    excel = workbook.metadata.format.excel if workbook.metadata.format else None
    if excel is None or excel.sheet_names is None:
        msg = "xberg found no sheets in the workbook."
        raise ValueError(msg)
    # A table's page is its sheet's position; xberg skips empty sheets.
    cells: dict[int, list[list[str]]] = {}
    for table in workbook.tables:
        cells.setdefault(table.page_number, []).extend(table.cells)
    return [
        (name, cells.get(page_number, []))
        for page_number, name in enumerate(excel.sheet_names, start=1)
    ]
