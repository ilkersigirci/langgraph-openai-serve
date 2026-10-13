import httpx2
import pytest
from langchain_core.runnables import RunnableLambda

from lgos_demo_api.utils.models import attribute_end_user


@pytest.mark.parametrize(
    ("metadata", "end_user"),
    [({"lgos.user": "user-123"}, "user-123"), ({}, None)],
    ids=["run-with-user", "run-without-user"],
)
async def test_gateway_calls_in_a_graph_run_bill_its_end_user(
    metadata: dict[str, str], end_user: str | None
) -> None:
    request = httpx2.Request("POST", "http://gateway/v1/chat/completions")

    async def call_gateway(_: object) -> None:
        await attribute_end_user(request)

    await RunnableLambda(call_gateway).ainvoke(None, config={"metadata": metadata})

    assert request.headers.get("x-litellm-end-user-id") == end_user


async def test_gateway_calls_outside_a_graph_run_bill_no_end_user() -> None:
    request = httpx2.Request("POST", "http://gateway/v1/chat/completions")

    await attribute_end_user(request)

    assert "x-litellm-end-user-id" not in request.headers
