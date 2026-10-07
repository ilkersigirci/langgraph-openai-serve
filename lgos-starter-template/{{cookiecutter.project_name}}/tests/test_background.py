from uuid import uuid4

import anyio
from openai import AsyncOpenAI
from openai.types.responses import Response

from tests.support import ANSWER, create_test_app, started


async def finished(client: AsyncOpenAI, response: Response) -> Response:
    with anyio.fail_after(5):
        while response.status in {"queued", "in_progress"}:
            await anyio.lowlevel.checkpoint()
            response = await client.responses.retrieve(response.id)
    return response


async def test_background_submission_polling_and_idempotency() -> None:
    async with started(create_test_app(background="memory")) as client:
        headers = {"Idempotency-Key": str(uuid4())}
        accepted = await client.responses.create(
            model="simple-graph",
            input="Hello",
            background=True,
            store=True,
            extra_headers=headers,
        )
        replayed = await client.responses.create(
            model="simple-graph",
            input="Hello",
            background=True,
            store=True,
            extra_headers=headers,
        )
        assert replayed.id == accepted.id
        response = await finished(client, accepted)
    assert response.status == "completed"
    assert response.output_text == ANSWER


async def test_approval_pauses_in_the_background() -> None:
    async with started(create_test_app(background="memory")) as client:
        accepted = await client.responses.create(
            model="approval", input="Publish the report", background=True, store=True
        )
        paused = await finished(client, accepted)
    assert paused.status == "completed"
    calls = [item for item in paused.output if item.type == "function_call"]
    assert [call.name for call in calls] == ["lgos_interrupt"]
