import json
import os
import uuid

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError
from openai.types.responses import ResponseFunctionToolCall

LITELLM_BASE_URL = os.getenv("DEMO_TEST_LITELLM_BASE_URL")
LITELLM_CATALOG_BASE_URL = os.getenv("DEMO_TEST_LITELLM_CATALOG_BASE_URL")
LITELLM_API_KEY = os.getenv(
    "DEMO_TEST_LITELLM_API_KEY",
    "sk-lgos-litellm-demo",
)
FILES_QUERY = {"provider": "litellm_proxy"}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        LITELLM_BASE_URL is None,
        reason="set the native LiteLLM test URL",
    ),
]


async def test_litellm_admin_ui_login() -> None:
    assert LITELLM_BASE_URL is not None

    async with httpx.AsyncClient(
        base_url=LITELLM_BASE_URL.removesuffix("/v1"),
        timeout=10.0,
    ) as client:
        response = await client.post(
            "/v2/login",
            json={"username": "admin", "password": LITELLM_API_KEY},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["redirect_url"].endswith("/ui?login=success")
    assert body["token"]


async def test_litellm_passthrough_requires_gateway_authentication() -> None:
    if LITELLM_CATALOG_BASE_URL is None:
        pytest.skip("set the LiteLLM catalog root URL")

    async with httpx.AsyncClient(
        base_url=f"{LITELLM_CATALOG_BASE_URL}/lgos-a",
        timeout=10.0,
    ) as client:
        response = await client.get("/models")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "auth_error"


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_litellm_ui_catalog_drives_managed_responses(provider: str) -> None:
    if LITELLM_CATALOG_BASE_URL is None:
        pytest.skip("set the LiteLLM catalog root URL")
    assert LITELLM_BASE_URL is not None

    async with (
        AsyncOpenAI(
            base_url=f"{LITELLM_CATALOG_BASE_URL}/{provider}",
            api_key=LITELLM_API_KEY,
            max_retries=0,
            timeout=10.0,
        ) as catalog_client,
        AsyncOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            max_retries=0,
            timeout=10.0,
        ) as responses_client,
    ):
        catalog = await catalog_client.models.list()
        model = next(
            item for item in catalog.data if item.id == "custom-input-output-context"
        )
        detail = await catalog_client.models.retrieve(model.id)
        response = await responses_client.responses.create(
            model=f"{provider}/{model.id}",
            input="Use the catalog model through managed routing.",
            store=False,
            user="gateway-user",
        )

    extension = (detail.model_extra or {})["langgraph_openai_serve"]
    assert extension["description"]
    assert response.output_text == (
        "gateway-user asked: Use the catalog model through managed routing."
    )


async def test_litellm_files_route_preserves_content() -> None:
    assert LITELLM_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        uploaded = await client.files.create(
            file=("attachment.bin", b"demo attachment"),
            purpose="user_data",
            extra_query=FILES_QUERY,
        )
        try:
            metadata = await client.files.retrieve(
                uploaded.id,
                extra_query=FILES_QUERY,
            )
            content = await client.files.content(
                uploaded.id,
                extra_query=FILES_QUERY,
            )
            assert metadata.filename == "attachment.bin"
            assert await content.aread() == b"demo attachment"
        finally:
            deleted = await client.files.delete(
                uploaded.id,
                extra_query=FILES_QUERY,
            )
            assert deleted.deleted is True


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_litellm_native_responses_preserve_file_input(provider: str) -> None:
    assert LITELLM_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        uploaded = await client.files.create(
            file=("attachment.bin", b"demo attachment"),
            purpose="user_data",
            extra_query=FILES_QUERY,
        )
        try:
            response = await client.responses.create(
                model=f"{provider}/custom-input-output-context",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Use this file."},
                            {"type": "input_file", "file_id": uploaded.id},
                        ],
                    }
                ],
                store=False,
                user="gateway-user",
            )
            assert response.output_text.startswith("gateway-user asked:")
            assert uploaded.id in response.output_text
        finally:
            deleted = await client.files.delete(
                uploaded.id,
                extra_query=FILES_QUERY,
            )
            assert deleted.deleted is True


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_litellm_native_responses_preserve_lgos_output(provider: str) -> None:
    assert LITELLM_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        response = await client.responses.create(
            model=f"{provider}/custom-input-output-context",
            input="Where is the routing boundary?",
            store=False,
            user="gateway-user",
        )
        assert response.output_text == (
            "gateway-user asked: Where is the routing boundary?"
        )
        assert response.store is False
        assert response.output[0].phase == "final_answer"

        stream = await client.responses.create(
            model=f"{provider}/custom-input-output-context",
            input="Stream through the gateway.",
            store=False,
            user="gateway-user",
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
    assert completed[0].response.output_text == (
        "gateway-user asked: Stream through the gateway."
    )


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_litellm_native_stream_preserves_commentary(provider: str) -> None:
    assert LITELLM_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        stream = await client.responses.create(
            model=f"{provider}/status-events",
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


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_litellm_native_function_output_continuation(provider: str) -> None:
    assert LITELLM_BASE_URL is not None
    model = f"{provider}/interruptible-approval"
    public_request = f"Refund order ORDER-{provider.upper()}"

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        paused = await client.responses.create(
            model=model,
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
            model=model,
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


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="LiteLLM v1.99.1 rewrites upstream OpenAI error metadata",
)
async def test_litellm_preserves_openai_errors(provider: str) -> None:
    assert LITELLM_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        with pytest.raises(BadRequestError) as exc_info:
            await client.responses.create(
                model=f"{provider}/missing-gateway-model",
                input="Hi",
            )

    assert exc_info.value.response.status_code == 400
    error = exc_info.value.response.json()["error"]
    assert (error.get("type"), error.get("param"), error.get("code")) == (
        "invalid_request_error",
        "model",
        None,
    )
