import pytest
from anyio import create_task_group, fail_after
from fastapi import FastAPI
from openai import AsyncOpenAI, ConflictError
from openai.types.chat import ChatCompletion

from .support import (
    CONCURRENT_MODEL,
    create_completion,
    resume_messages,
)


async def test_concurrent_resume_executes_post_interrupt_work_once(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    first_response = await create_completion(openai_client, model=CONCURRENT_MODEL)
    messages = resume_messages(first_response, ["approve"])

    responses: list[ChatCompletion] = []

    async def complete_first_resume() -> None:
        response = await openai_client.chat.completions.create(
            model=CONCURRENT_MODEL,
            messages=messages,
        )
        responses.append(response)

    async with create_task_group() as task_group:
        task_group.start_soon(complete_first_resume)
        with fail_after(1):
            await fastapi_app.state.resume_entered.wait()
        try:
            with fail_after(1), pytest.raises(ConflictError) as exc_info:
                await openai_client.with_options(max_retries=0).chat.completions.create(
                    model=CONCURRENT_MODEL,
                    messages=messages,
                )
        finally:
            fastapi_app.state.resume_release.set()

    assert len(responses) == 1
    assert responses[0].choices[0].message.content == "approve"
    assert fastapi_app.state.side_effects == {"count": 1}
    assert exc_info.value.body["code"] == "run_busy"
