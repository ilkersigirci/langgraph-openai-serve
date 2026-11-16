"""Shared live Files gateway assertions."""

from openai import AsyncOpenAI

NOTES = b"# Notes\n\nShip it."


async def assert_document_extraction_contract(
    base_url: str, api_key: str, provider: str
) -> None:
    """
    Read a document's extracted text as Open WebUI's Files API support does.

    The image waits for the file's status and then reads its content, so the
    gateway must pass both through unchanged.
    """
    async with AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_query={"provider": provider},
        max_retries=0,
        timeout=10.0,
    ) as client:
        uploaded = await client.files.create(
            file=("notes.md", NOTES), purpose="user_data"
        )
        try:
            processed = await client.files.wait_for_processing(
                uploaded.id, poll_interval=0.2, max_wait_seconds=30
            )
            content = await client.files.content(uploaded.id)
            # The Files API extracts after the upload returns.
            assert (uploaded.status, processed.status) == ("uploaded", "processed")
            assert content.json() == {
                "type": "text",
                "version": 1,
                "text": NOTES.decode(),
            }
        finally:
            deleted = await client.files.delete(uploaded.id)
            assert deleted.deleted is True
