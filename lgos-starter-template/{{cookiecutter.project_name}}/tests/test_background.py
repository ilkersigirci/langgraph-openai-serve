from uuid import uuid4

import anyio

from tests.support import ANSWER, create_test_app, started


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
        response = accepted
        with anyio.fail_after(5):
            while response.status in {"queued", "in_progress"}:
                await anyio.lowlevel.checkpoint()
                response = await client.responses.retrieve(accepted.id)
    assert response.status == "completed"
    assert response.output_text == ANSWER
