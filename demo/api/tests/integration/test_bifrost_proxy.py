import json
import os

import pytest
from openai import AsyncOpenAI, BadRequestError
from openai.types.responses import ResponseFunctionToolCall
from tests.integration.mcp_gateway import assert_postgres_mcp_contract

BIFROST_BASE_URL = os.getenv("DEMO_TEST_BIFROST_BASE_URL")
BIFROST_CATALOG_BASE_URL = os.getenv("DEMO_TEST_BIFROST_CATALOG_BASE_URL")
BIFROST_API_KEY = os.getenv("OPENAI_GATEWAY_API_KEY", "sk-bf-lgos-demo-only-key")

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
    reason="Bifrost normalized model detail omits LGOS extensions",
)


async def test_bifrost_native_mcp_is_authenticated_and_exposes_fixed_reports() -> None:
    assert BIFROST_CATALOG_BASE_URL is not None

    await assert_postgres_mcp_contract(
        BIFROST_CATALOG_BASE_URL.removesuffix("/v1"),
        BIFROST_API_KEY,
        endpoint="/mcp",
    )


async def test_bifrost_catalog_and_files_preserve_lgos() -> None:
    assert BIFROST_BASE_URL is not None
    assert BIFROST_CATALOG_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_CATALOG_BASE_URL,
        api_key=BIFROST_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as catalog:
        catalog_models = await catalog.models.list()
        models = {
            model.id: model
            for model in catalog_models.data
            if model.owned_by == "langgraph-openai-serve"
        }
        for model_id in ("lgos-a/simple-graph", "lgos-b/simple-graph"):
            # The catalog sync publishes the complete detail extension, which
            # the normalized /openai/v1 model routes omit.
            attributes = (models[model_id].model_extra or {})["additional_attributes"]
            extension = json.loads(attributes["lgos"])
            assert set(extension["client_settings"]) == {"defaults", "json_schema"}

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
            api_key=BIFROST_API_KEY,
            max_retries=0,
            timeout=10.0,
        ) as catalog,
        AsyncOpenAI(
            base_url=BIFROST_BASE_URL,
            api_key=BIFROST_API_KEY,
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
        api_key=BIFROST_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        model = await client.models.retrieve(f"{provider}/simple-graph")

    model_extra = getattr(model, "model_extra", None)
    assert isinstance(model_extra, dict)
    extension = model_extra["lgos"]
    assert set(extension["client_settings"]) == {"defaults", "json_schema"}


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_bifrost_native_responses_preserve_standard_fields(
    provider: str,
) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key=BIFROST_API_KEY,
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
    assert response.output[0].phase == "final_answer"
    assert (response.model_extra or {})["store"] is False


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_bifrost_native_stream_preserves_commentary(provider: str) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key=BIFROST_API_KEY,
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
    response_events = [
        event
        for event in events
        if event.type
        in {"response.created", "response.in_progress", "response.completed"}
    ]
    assert [event.type for event in response_events] == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]
    assert all(
        (event.response.model_extra or {})["store"] is False
        for event in response_events
    )


@pytest.mark.parametrize("provider", ["lgos-a", "lgos-b"])
async def test_bifrost_native_function_output_continuation(provider: str) -> None:
    assert BIFROST_BASE_URL is not None
    model = f"{provider}/interruptible-approval"
    public_request = f"Refund order ORDER-{provider.upper()}"

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key=BIFROST_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        paused = await client.responses.create(
            model=model,
            input=public_request,
            store=False,
        )
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
async def test_bifrost_preserves_openai_errors(provider: str) -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key=BIFROST_API_KEY,
        max_retries=0,
        timeout=10.0,
    ) as client:
        with pytest.raises(BadRequestError) as exc_info:
            # Governance rejects unknown provider-qualified models itself, so
            # request an LGOS validation error from a registered graph.
            await client.responses.create(
                model=f"{provider}/simple-graph",
                input="Hi",
                tools=[{"type": "custom", "name": "unknown_server_tool"}],
            )

    assert exc_info.value.response.status_code == 400
    error = exc_info.value.response.json()["error"]
    assert (error.get("type"), error.get("param"), error.get("code")) == (
        "invalid_request_error",
        "tools.0.name",
        None,
    )
