import json
import os
import uuid

import pytest
from openai import AsyncOpenAI, BadRequestError
from openai.types.responses import ResponseFunctionToolCall

DIRECT_BASE_URLS = tuple(
    base_url.strip()
    for base_url in os.getenv("DEMO_TEST_DIRECT_BASE_URLS", "").split(",")
    if base_url.strip()
)
FILES_BASE_URL = os.getenv("DEMO_TEST_FILES_BASE_URL")
API_KEY = os.getenv("DEMO_TEST_OPENAI_API_KEY", "DUMMY")
MODEL_PROVIDER = os.getenv("DEMO_TEST_OPENAI_MODEL_PROVIDER")
FILES_PROVIDER = os.getenv("DEMO_TEST_FILES_PROVIDER")
ENDPOINTS = DIRECT_BASE_URLS or (None,)


def _graph_client(base_url: str) -> AsyncOpenAI:
    default_headers = (
        {"x-model-provider": MODEL_PROVIDER} if MODEL_PROVIDER is not None else None
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key=API_KEY,
        max_retries=0,
        timeout=10.0,
        default_headers=default_headers,
    )


def _files_query() -> dict[str, str]:
    return {"provider": FILES_PROVIDER} if FILES_PROVIDER is not None else {}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DIRECT_BASE_URLS,
        reason="set the comma-separated direct LGOS test URLs",
    ),
]


async def test_direct_files_preserve_content() -> None:
    if FILES_BASE_URL is None:
        pytest.skip("set the direct Files test URL")

    async with AsyncOpenAI(
        base_url=FILES_BASE_URL,
        api_key=API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        uploaded = await client.files.create(
            file=("attachment.bin", b"demo attachment"),
            purpose="user_data",
            extra_query=_files_query(),
        )
        try:
            metadata = await client.files.retrieve(
                uploaded.id,
                extra_query=_files_query(),
            )
            content = await client.files.content(
                uploaded.id,
                extra_query=_files_query(),
            )
            assert metadata.filename == "attachment.bin"
            assert await content.aread() == b"demo attachment"
        finally:
            deleted = await client.files.delete(
                uploaded.id,
                extra_query=_files_query(),
            )
            assert deleted.deleted is True


@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_direct_model_catalog_preserves_lgos_metadata(
    base_url: str | None,
) -> None:
    assert base_url is not None

    async with _graph_client(base_url) as client:
        models = await client.models.list()
        model = await client.models.retrieve("simple-graph")

    assert any(item.id == "simple-graph" for item in models.data)
    extension = (model.model_extra or {})["langgraph_openai_serve"]
    assert extension["schema_version"] == 1
    assert isinstance(extension["description"], str)


@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_direct_responses_preserve_file_input(
    base_url: str | None,
) -> None:
    assert base_url is not None
    file_id = "file-direct-transport"

    async with _graph_client(base_url) as client:
        response = await client.responses.create(
            model="custom-input-output-context",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Use this file."},
                        {"type": "input_file", "file_id": file_id},
                    ],
                }
            ],
            store=False,
            user="direct-user",
        )

    assert response.output_text.startswith("direct-user asked:")
    assert file_id in response.output_text


@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_direct_responses_preserve_text_and_stream(
    base_url: str | None,
) -> None:
    assert base_url is not None

    async with _graph_client(base_url) as client:
        response = await client.responses.create(
            model="custom-input-output-context",
            input="Where is the routing boundary?",
            store=False,
            user="direct-user",
        )
        assert response.output_text == (
            "direct-user asked: Where is the routing boundary?"
        )
        assert response.store is False
        assert response.output[0].phase == "final_answer"

        stream = await client.responses.create(
            model="custom-input-output-context",
            input="Stream directly.",
            store=False,
            user="direct-user",
            stream=True,
        )
        events = [event async for event in stream]

    added_items = [
        event.item for event in events if event.type == "response.output_item.added"
    ]
    assert [(item.type, item.phase) for item in added_items] == [
        ("message", "final_answer")
    ]
    completed = [event for event in events if event.type == "response.completed"]
    assert len(completed) == 1
    assert completed[0].response.output_text == "direct-user asked: Stream directly."


@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_direct_stream_preserves_commentary(base_url: str | None) -> None:
    assert base_url is not None

    async with _graph_client(base_url) as client:
        stream = await client.responses.create(
            model="status-events",
            input="Build the report.",
            store=False,
            stream=True,
        )
        events = [event async for event in stream]

    added_items = [
        event.item for event in events if event.type == "response.output_item.added"
    ]
    assert [(item.phase, item.type) for item in added_items] == [
        ("commentary", "message"),
        ("commentary", "message"),
        ("commentary", "message"),
        ("final_answer", "message"),
    ]


@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_direct_function_output_continuation(base_url: str | None) -> None:
    assert base_url is not None
    public_request = "Refund order ORDER-DIRECT"

    async with _graph_client(base_url) as client:
        paused = await client.responses.create(
            model="interruptible-approval",
            input=public_request,
            metadata={"langgraph_run_id": str(uuid.uuid4())},
            store=False,
        )
        assert len(paused.output) == 1
        call = paused.output[0]
        assert isinstance(call, ResponseFunctionToolCall)
        arguments = json.loads(call.arguments)
        assert arguments["payload"]["action"] == "refund"

        completed = await client.responses.create(
            model="interruptible-approval",
            input=[
                *paused.output,
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps({"resume": "approve"}),
                },
            ],
            store=False,
        )

    assert completed.output_text == (
        f"Review workflow for: {public_request}\n"
        "- Refund: approve\n"
        "- Customer notification: sent\n"
        "- Executed actions: Refund, Customer notification"
    )


@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_direct_responses_preserve_openai_errors(base_url: str | None) -> None:
    assert base_url is not None

    async with _graph_client(base_url) as client:
        with pytest.raises(BadRequestError) as exc_info:
            await client.responses.create(model="missing-gateway-model", input="Hi")

    assert exc_info.value.response.status_code == 400
    error = exc_info.value.response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == (
        "invalid_request_error",
        "model",
        None,
    )
