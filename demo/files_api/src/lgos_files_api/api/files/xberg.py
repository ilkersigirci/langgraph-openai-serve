"""
PDF and Office documents, extracted with xberg in this process.

xberg reads their text layer and OCRs pages without one. Their content is a
text content (native.py) holding the document's Markdown.
"""

from pathlib import PurePath

import xberg

from lgos_files_api.api.files.native import text_content

XBERG_SUFFIXES = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}
# The container image bundles English OCR data only; xberg would download the
# data of other languages on first use.
OCR_LANGUAGES = ["eng"]


def is_supported(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in XBERG_SUFFIXES


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
            xberg.ExtractInput(kind="bytes", bytes=document, filename=filename),  # ty: ignore[invalid-argument-type] - xberg 1.3 annotates the field with its own name, which ty reads as None.
            config,
        )
    # Whatever xberg raises means it can't read this document: xberg 1.3
    # raises RuntimeError, not its XbergError subclasses, for unreadable
    # documents and timeouts alike, and its async bridge reports a Rust panic as
    # its own exception.
    except Exception as error:
        if isinstance(error, xberg.XbergTimeoutError) or str(error).startswith(
            "Extraction timed out"
        ):
            raise TimeoutError(str(error)) from error
        raise ValueError(str(error)) from error
    if result.errors:
        raise ValueError(result.errors[0].message)
    return text_content("\n\n".join(extracted.content for extracted in result.results))
