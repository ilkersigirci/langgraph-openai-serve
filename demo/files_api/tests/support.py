"""Documents and clients for the Files service tests."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
from xml.sax.saxutils import escape
from zipfile import ZipFile

from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from openai import AsyncOpenAI
from openai.types import FileObject
from openpyxl import Workbook

BUCKET = "files"

_WORD_PARTS = {
    "[Content_Types].xml": (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    ),
    "_rels/.rels": (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    ),
}


def docx(*paragraphs: str) -> bytes:
    """Return a Word document with these paragraphs, the smallest one Word opens."""
    body = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, xml in _WORD_PARTS.items():
            archive.writestr(name, xml)
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>",
        )
    return buffer.getvalue()


def xlsx(title: str, rows: list[list[object]]) -> bytes:
    """Return a workbook with one sheet of these rows."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = title
    for row in rows:
        sheet.append(row)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncGenerator[AsyncOpenAI]:
    """Serve the app, with its background extractions, to an OpenAI client."""
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http,
            max_retries=0,
        ) as client,
    ):
        yield client


async def processed_file(client: AsyncOpenAI, file_id: str) -> FileObject:
    """Wait, as SDK clients do, until the file's extraction has ended."""
    return await client.files.wait_for_processing(
        file_id, poll_interval=0.01, max_wait_seconds=30
    )
