"""Contract tests for interrupt execution and API rejection."""

import uuid
from http import HTTPStatus

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, BadRequestError, InternalServerError
from openai.types.responses import ResponseFunctionToolCall

from tests.graph.support.interrupt import DEFAULT_INTERRUPT_PAYLOAD

from .support import (
    INVALID_PAYLOAD_MODEL,
    MODEL,
    NESTED_MODEL,
    PARALLEL_MODEL,
    assert_checkpoint_deleted,
    assert_interrupt_arguments,
    create_response,
    interrupt_calls,
    resume_outputs,
    resume_response,
)

EXPECTED_PARALLEL_INTERRUPTS = 2


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_completions_rejects_interrupt_graphs(
    openai_client: AsyncOpenAI,
    stream: bool,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            stream=stream,
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert "requires interrupts, which is only supported via the Responses API" in str(
        exc_info.value
    )
    assert exc_info.value.body["param"] == "model"


async def test_non_streaming_interrupt_matches_contract_and_resumes(
    openai_client: AsyncOpenAI,
) -> None:
    response = await create_response(openai_client)

    calls = interrupt_calls(response)
    assert len(calls) == 1
    arguments = assert_interrupt_arguments(calls[0])
    assert arguments == DEFAULT_INTERRUPT_PAYLOAD

    final_response = await resume_response(openai_client, response, "approve")
    assert final_response.output_text == "resumed:approve"
    assert final_response.previous_response_id == response.id


async def test_interrupt_output_requires_previous_response_id(
    openai_client: AsyncOpenAI,
) -> None:
    first = await create_response(openai_client)
    call = interrupt_calls(first)[0]

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            input=[
                *first.output,
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "approve",
                },
            ],
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert "require previous_response_id" in str(exc_info.value)


async def test_invalid_interrupt_response_id_reports_its_parameter(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            previous_response_id="resp_invalid",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_invalid",
                    "output": "approve",
                }
            ],
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "previous_response_id"


async def test_interrupt_resume_rejects_new_instructions(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            previous_response_id=first_response.id,
            input=resume_outputs(first_response, ["approve"]),
            instructions="Use a different policy.",
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "instructions"


async def test_function_output_is_passed_to_langgraph_as_a_string(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)
    final_response = await resume_response(openai_client, first_response, None)

    assert final_response.output_text == "resumed:null"


async def test_streaming_interrupt_matches_responses_tool_call_contract(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await create_response(openai_client, stream=True)
    events = [event async for event in stream]

    added = [e for e in events if e.type == "response.output_item.added"]
    assert len(added) == 1
    assert added[0].item.type == "function_call"

    delta = [e for e in events if e.type == "response.function_call_arguments.delta"]
    assert len(delta) == 1

    done = [e for e in events if e.type == "response.output_item.done"]
    assert len(done) == 1
    assert isinstance(done[0].item, ResponseFunctionToolCall)
    arguments = assert_interrupt_arguments(done[0].item)
    assert arguments == DEFAULT_INTERRUPT_PAYLOAD
    assert events[-1].type == "response.completed"
    assert events[-1].response.status == "completed"
    response_events = [event.response for event in events if hasattr(event, "response")]
    assert len({response.id for response in response_events}) == 1


async def test_invalid_interrupt_payload_returns_openai_server_error(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    run_id = str(uuid.uuid4())
    with pytest.raises(InternalServerError) as exc_info:
        await create_response(
            openai_client,
            model=INVALID_PAYLOAD_MODEL,
            run_id=run_id,
        )

    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert exc_info.value.body == {
        "message": "LangGraph interrupt payloads must be valid JSON values.",
        "type": "server_error",
        "param": None,
        "code": None,
    }
    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
    )


async def test_streaming_invalid_interrupt_payload_deletes_checkpoint(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    run_id = str(uuid.uuid4())
    stream = await create_response(
        openai_client,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
        stream=True,
    )

    events = [event async for event in stream]
    assert [event.type for event in events][-2:] == ["error", "response.failed"]
    assert events[-1].response.status == "failed"

    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
    )


async def test_parallel_interrupts_are_one_tool_call_batch_and_resume_by_id(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client, model=PARALLEL_MODEL)
    calls = interrupt_calls(first_response)

    assert len(calls) == EXPECTED_PARALLEL_INTERRUPTS
    arguments = [assert_interrupt_arguments(call) for call in calls]
    assert {item["question"] for item in arguments} == {"left", "right"}

    final_response = await resume_response(
        openai_client,
        first_response,
        "first",
        "second",
        model=PARALLEL_MODEL,
    )
    assert final_response.output_text == "first,second"


async def test_parallel_nested_interrupts_resume_as_one_tool_call_batch(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client, model=NESTED_MODEL)
    calls = interrupt_calls(first_response)

    assert len(calls) == EXPECTED_PARALLEL_INTERRUPTS
    arguments = [assert_interrupt_arguments(call) for call in calls]
    assert {item["question"] for item in arguments} == {
        "nested-a",
        "nested-b",
    }
    values = [
        "first" if item["question"] == "nested-a" else "second" for item in arguments
    ]

    final_response = await resume_response(
        openai_client,
        first_response,
        *values,
        model=NESTED_MODEL,
    )
    assert final_response.output_text == "nested-a:first,nested-b:second"
