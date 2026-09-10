import json
import os
import uuid

import httpx
import pytest
from openai import AsyncOpenAI, AsyncStream, BadRequestError
from openai.types.responses import ResponseFunctionToolCall

LITELLM_BASE_URL = os.getenv("DEMO_TEST_LITELLM_BASE_URL")
DIRECT_BASE_URLS = os.getenv("DEMO_TEST_DIRECT_BASE_URLS", "").split(",")
LITELLM_API_KEY = os.getenv("DEMO_TEST_LITELLM_API_KEY") or os.getenv(
    "OPENAI_GATEWAY_API_KEY", ""
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


async def test_litellm_chat_catalog_discovers_lgos_models() -> None:
    assert LITELLM_BASE_URL is not None

    async with httpx.AsyncClient(
        base_url=LITELLM_BASE_URL.removesuffix("/v1"),
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        timeout=10.0,
    ) as client:
        response = await client.get("/model_group/info")

    assert response.status_code == 200
    model_groups = {item["model_group"] for item in response.json()["data"]}
    assert {"lgos-a/simple-graph", "lgos-b/simple-graph"} <= model_groups


async def test_litellm_model_info_requires_gateway_authentication() -> None:
    assert LITELLM_BASE_URL is not None
    async with httpx.AsyncClient(
        base_url=LITELLM_BASE_URL.removesuffix("/v1"),
        timeout=10.0,
    ) as client:
        response = await client.get("/model/info")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "auth_error"


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_litellm_ui_catalog_drives_managed_responses(provider: str) -> None:
    assert LITELLM_BASE_URL is not None
    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        catalog = await client.get(
            f"{LITELLM_BASE_URL.removesuffix('/v1')}/model/info",
            cast_to=object,
        )
        model = next(
            item
            for item in catalog["data"]
            if item["model_name"] == f"{provider}/custom-input-output-context"
        )
        extension = model["model_info"]["lgos"]
        assert extension["schema_version"] == 1
        assert extension["description"]
        response = await client.responses.create(
            model=model["model_name"],
            input="Use the catalog model through managed routing.",
            store=False,
            user="gateway-user",
        )

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
@pytest.mark.parametrize(
    ("model", "prompt", "commentary_count"),
    [
        ("status-events", "Build the report.", 3),
        ("complex-subgraphs", "Show nested subgraph routing docs.", 1),
    ],
)
async def test_litellm_native_stream_preserves_commentary(
    provider: str, model: str, prompt: str, commentary_count: int
) -> None:
    assert LITELLM_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        stream = await client.responses.create(
            model=f"{provider}/{model}",
            input=prompt,
            store=False,
            stream=True,
        )
        events = [event async for event in stream]

    added_items = [
        event.item for event in events if event.type == "response.output_item.added"
    ]
    assert [(item.phase, item.type) for item in added_items] == [
        *[("commentary", "message")] * commentary_count,
        ("final_answer", "message"),
    ]


@pytest.mark.parametrize("provider,source_index", [("lgos-a", 0), ("lgos-b", 1)])
async def test_litellm_preserves_upstream_text_deltas(
    provider: str,
    source_index: int,
) -> None:
    if (
        len(DIRECT_BASE_URLS) <= source_index
        or not DIRECT_BASE_URLS[source_index].strip()
    ):
        pytest.skip("set the comma-separated direct LGOS test URLs")
    assert LITELLM_BASE_URL is not None

    deltas_by_route: list[list[str]] = []
    for base_url, model, api_key in (
        (
            DIRECT_BASE_URLS[source_index].strip(),
            "multi-node-streaming",
            os.getenv("DEMO_TEST_OPENAI_API_KEY", "DUMMY"),
        ),
        (LITELLM_BASE_URL, f"{provider}/multi-node-streaming", LITELLM_API_KEY),
    ):
        async with AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=10.0,
        ) as client:
            stream = await client.responses.create(
                model=model,
                input="Combine both contributions.",
                store=False,
                stream=True,
            )
            async with stream:
                deltas_by_route.append(
                    [
                        event.delta
                        async for event in stream
                        if event.type == "response.output_text.delta"
                    ]
                )

    upstream, managed = deltas_by_route
    assert len(upstream) > 1
    assert "".join(upstream) == (
        "The first node contributed this sentence. "
        "The second node contributed this sentence."
    )
    # A synthetic stream can contain many deltas yet deliver them only after
    # generation. It must preserve the upstream chunks, not split final text.
    assert managed == upstream


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_litellm_native_function_output_continuation(
    provider: str, stream: bool
) -> None:
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
            metadata={"lgos_run_id": str(uuid.uuid4())},
            store=False,
            stream=stream,
        )
        if isinstance(paused, AsyncStream):
            async with paused:
                completed_events = [
                    event
                    async for event in paused
                    if event.type == "response.completed"
                ]
            assert len(completed_events) == 1
            paused = completed_events[0].response
        assert len(paused.output) == 1
        call = paused.output[0]
        assert isinstance(call, ResponseFunctionToolCall)
        arguments = json.loads(call.arguments)
        assert arguments["action"] == "refund"

        completed = await client.responses.create(
            model=model,
            previous_response_id=paused.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "approve",
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
    reason="LiteLLM v1.100.0 rewrites upstream OpenAI error metadata",
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
