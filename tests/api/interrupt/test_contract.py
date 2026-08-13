import uuid
from http import HTTPStatus

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import APIError, AsyncOpenAI, InternalServerError

from tests.graph.support.interrupt import DEFAULT_INTERRUPT_PAYLOAD

from .support import (
    INVALID_PAYLOAD_MODEL,
    NESTED_MODEL,
    PARALLEL_MODEL,
    assert_checkpoint_deleted,
    assert_interrupt_arguments,
    create_completion,
    resume_interrupt,
)

EXPECTED_PARALLEL_INTERRUPTS = 2


async def test_non_streaming_interrupt_matches_contract_and_resumes(
    openai_client: AsyncOpenAI,
) -> None:
    response = await create_completion(openai_client)

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    arguments = assert_interrupt_arguments(choice.message.tool_calls[0])
    assert arguments["payload"] == DEFAULT_INTERRUPT_PAYLOAD
    final_response = await resume_interrupt(openai_client, response, "approve")

    assert final_response.choices[0].message.content == "resumed:approve"


async def test_id_mapped_json_null_is_a_valid_resume_value(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client)
    final_response = await resume_interrupt(openai_client, first_response, None)

    assert final_response.choices[0].message.content == "resumed:None"


async def test_streaming_interrupt_matches_openai_tool_call_contract(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await create_completion(openai_client, stream=True)
    chunks = [chunk async for chunk in stream]

    tool_call_chunks = [
        chunk for chunk in chunks if chunk.choices[0].delta.tool_calls is not None
    ]
    assert len(tool_call_chunks) == 1
    tool_call = tool_call_chunks[0].choices[0].delta.tool_calls[0]
    assert tool_call.index == 0
    assert_interrupt_arguments(tool_call)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


async def test_invalid_interrupt_payload_returns_openai_server_error(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    run_id = str(uuid.uuid4())
    with pytest.raises(InternalServerError) as exc_info:
        await create_completion(
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
    stream = await create_completion(
        openai_client,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
        stream=True,
    )

    with pytest.raises(APIError, match="Internal server error"):
        _ = [chunk async for chunk in stream]

    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
    )


async def test_parallel_interrupts_are_one_tool_call_batch_and_resume_by_id(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client, model=PARALLEL_MODEL)
    tool_calls = first_response.choices[0].message.tool_calls or []

    assert len(tool_calls) == EXPECTED_PARALLEL_INTERRUPTS
    arguments = [assert_interrupt_arguments(tool_call) for tool_call in tool_calls]
    assert {item["payload"]["question"] for item in arguments} == {"left", "right"}
    assert len({item["state_token"] for item in arguments}) == 1
    assert len({item["run_id"] for item in arguments}) == 1

    final_response = await resume_interrupt(
        openai_client,
        first_response,
        "first",
        "second",
        model=PARALLEL_MODEL,
    )
    assert final_response.choices[0].message.content == "first,second"


async def test_parallel_nested_interrupts_resume_as_one_tool_call_batch(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client, model=NESTED_MODEL)
    tool_calls = first_response.choices[0].message.tool_calls or []

    assert len(tool_calls) == EXPECTED_PARALLEL_INTERRUPTS
    arguments = [assert_interrupt_arguments(tool_call) for tool_call in tool_calls]
    assert {item["payload"]["question"] for item in arguments} == {
        "nested-a",
        "nested-b",
    }
    values = [
        "first" if item["payload"]["question"] == "nested-a" else "second"
        for item in arguments
    ]

    final_response = await resume_interrupt(
        openai_client,
        first_response,
        *values,
        model=NESTED_MODEL,
    )

    assert final_response.choices[0].message.content == (
        "nested-a:first,nested-b:second"
    )
