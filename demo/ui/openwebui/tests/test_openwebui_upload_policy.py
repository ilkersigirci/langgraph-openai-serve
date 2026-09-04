from urllib.parse import parse_qsl

from lgos_openwebui.upload_policy import (
    Message,
    RawFileUploads,
    Receive,
    Scope,
    Send,
)


async def test_file_upload_policy_disables_processing_and_preserves_other_options() -> (
    None
):
    received_scope: Scope | None = None

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal received_scope
        received_scope = scope

    app = RawFileUploads(downstream)
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/files/",
        "query_string": b"process=true&process_in_background=false",
    }

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    await app(scope, receive, send)

    assert received_scope is not None
    query_string = received_scope["query_string"]
    assert isinstance(query_string, bytes)
    assert parse_qsl(query_string.decode("ascii")) == [
        ("process_in_background", "false"),
        ("process", "false"),
    ]
