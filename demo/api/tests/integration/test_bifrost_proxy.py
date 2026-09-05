import json
import os
import uuid

import pytest
from openai import AsyncOpenAI, BadRequestError
from openai.types.responses import ResponseFunctionToolCall

BIFROST_BASE_URL = os.getenv("DEMO_TEST_BIFROST_BASE_URL")
BIFROST_CATALOG_BASE_URL = os.getenv("DEMO_TEST_BIFROST_CATALOG_BASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        BIFROST_BASE_URL is None or BIFROST_CATALOG_BASE_URL is None,
        reason="set the native Bifrost and catalog test URLs",
    ),
]

BIFROST_MODEL_METADATA_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Bifrost v2.0.0 normalized model detail omits LGOS extensions",
)
BIFROST_ERROR_METADATA_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Bifrost v2.0.0 rewrites the upstream OpenAI error metadata",
)


async def test_bifrost_catalog_and_files_preserve_lgos() -> None:
    assert BIFROST_BASE_URL is not None
    assert BIFROST_CATALOG_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_CATALOG_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as catalog:
        catalog_models = await catalog.models.list()
        model_ids = {
            model.id
            for model in catalog_models.data
            if model.owned_by == "langgraph-openai-serve"
        }
        assert {"lgos-a/simple-graph", "lgos-b/simple-graph"} <= model_ids

        files_query = {"provider": "lgos-files"}
        uploaded = await catalog.files.create(
            file=("attachment.bin", b"demo attachment"),
            purpose="user_data",
            extra_query=files_query,
        )
        try:
            content = await catalog.files.content(
                uploaded.id,
                extra_query=files_query,
            )
            assert await content.aread() == b"demo attachment"
        finally:
            deleted = await catalog.files.delete(
                uploaded.id,
                extra_query=files_query,
            )
            assert deleted.deleted is True


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_bifrost_native_responses_preserve_file_input(provider: str) -> None:
    assert BIFROST_BASE_URL is not None
    assert BIFROST_CATALOG_BASE_URL is not None

    async with (
        AsyncOpenAI(
            base_url=BIFROST_CATALOG_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as catalog,
        AsyncOpenAI(
            base_url=BIFROST_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as api_client,
    ):
        files_query = {"provider": "lgos-files"}
        uploaded = await catalog.files.create(
            file=("attachment.bin", b"demo attachment"),
            purpose="user_data",
            extra_query=files_query,
        )
        try:
            response = await api_client.responses.create(
                model="custom-input-output-context",
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
                extra_headers={"x-model-provider": provider},
            )
            assert response.output_text.startswith("gateway-user asked:")
            assert uploaded.id in response.output_text
        finally:
            deleted = await catalog.files.delete(
                uploaded.id,
                extra_query=files_query,
            )
            assert deleted.deleted is True


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
@BIFROST_MODEL_METADATA_XFAIL
async def test_bifrost_native_route_preserves_model_metadata(provider: str) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as client:
        model = await client.models.retrieve(
            "simple-graph",
            extra_headers={"x-model-provider": provider},
        )

    model_extra = getattr(model, "model_extra", None)
    assert isinstance(model_extra, dict)
    extension = model_extra["langgraph_openai_serve"]
    assert extension["client_settings"]["schema_version"] == 1


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_bifrost_native_responses_preserve_standard_fields(
    provider: str,
) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as client:
        response = await client.responses.create(
            model="custom-input-output-context",
            input="Where is the routing boundary?",
            store=False,
            user="gateway-user",
            extra_headers={"x-model-provider": provider},
        )

    assert (
        response.output_text,
        response.store,
        response.output[0].phase,
    ) == (
        "gateway-user asked: Where is the routing boundary?",
        False,
        "final_answer",
    )


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_bifrost_native_stream_preserves_commentary(provider: str) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as client:
        stream = await client.responses.create(
            model="status-events",
            input="Build the report.",
            store=False,
            stream=True,
            extra_headers={"x-model-provider": provider},
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
async def test_bifrost_native_function_output_continuation(provider: str) -> None:
    assert BIFROST_BASE_URL is not None
    extra_headers = {"x-model-provider": provider}
    public_request = f"Refund order ORDER-{provider.upper()}"

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as client:
        paused = await client.responses.create(
            model="interruptible-approval",
            input=public_request,
            metadata={"langgraph_run_id": str(uuid.uuid4())},
            store=False,
            extra_headers=extra_headers,
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
            extra_headers=extra_headers,
        )

    assert completed.output_text == (
        f"Review workflow for: {public_request}\n"
        "- Refund: approve\n"
        "- Customer notification: sent\n"
        "- Executed actions: Refund, Customer notification"
    )


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
@BIFROST_ERROR_METADATA_XFAIL
async def test_bifrost_preserves_openai_errors(provider: str) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as client:
        with pytest.raises(BadRequestError) as exc_info:
            await client.responses.create(
                model="missing-gateway-model",
                input="Hi",
                extra_headers={"x-model-provider": provider},
            )

    assert exc_info.value.response.status_code == 400
    error = exc_info.value.response.json()["error"]
    assert (error.get("type"), error.get("param"), error.get("code")) == (
        "invalid_request_error",
        "model",
        None,
    )
