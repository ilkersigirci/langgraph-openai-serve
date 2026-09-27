from http import HTTPStatus

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, ConflictError

from langgraph_openai_serve.api.responses.interrupts import interrupt_tool_call_id

from .support import (
    MODEL,
    MULTI_TURN_MODEL,
    NESTED_MULTI_TURN_MODEL,
    assert_interrupt_arguments,
    assert_no_checkpoints,
    create_response,
    interrupt_calls,
    resume_outputs,
    resume_response,
)


async def test_paused_run_is_isolated_by_server_checkpoint_scope(
    openai_client: AsyncOpenAI,
) -> None:
    paused = await create_response(openai_client, checkpoint_scope="tenant-a")

    with pytest.raises(ConflictError):
        await resume_response(
            openai_client,
            paused,
            "approve",
            checkpoint_scope="tenant-b",
        )

    resumed = await resume_response(
        openai_client,
        paused,
        "approve",
        checkpoint_scope="tenant-a",
    )
    assert resumed.output_text == "resumed:approve"


async def test_fabricated_interrupt_id_cannot_resume_pending_state(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)
    input_items = resume_outputs(first_response, ["approve"])
    input_items[0]["call_id"] = interrupt_tool_call_id("fabricated")

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            previous_response_id=first_response.id,
            input=input_items,
        )

    assert exc_info.value.status_code == HTTPStatus.CONFLICT


@pytest.mark.parametrize("model", [MULTI_TURN_MODEL, NESTED_MULTI_TURN_MODEL])
async def test_native_interrupt_ids_reject_stale_answers_across_nodes(
    openai_client: AsyncOpenAI,
    model: str,
) -> None:
    first_pause = await create_response(openai_client, model=model)
    second_pause = await resume_response(
        openai_client,
        first_pause,
        "one",
        model=model,
    )

    first_call = interrupt_calls(first_pause)[0]
    second_call = interrupt_calls(second_pause)[0]
    assert first_call.call_id != second_call.call_id
    assert first_pause.id != second_pause.id
    first_arguments = assert_interrupt_arguments(first_call)
    second_arguments = assert_interrupt_arguments(second_call)
    assert first_arguments == {"question": "first"}
    assert second_arguments == {"question": "second"}

    with pytest.raises(ConflictError):
        await resume_response(
            openai_client,
            first_pause,
            "one",
            model=model,
        )

    final_response = await resume_response(
        openai_client,
        second_pause,
        "two",
        model=model,
    )
    assert final_response.output_text == "one,two"


async def test_streaming_state_conflict_returns_409_before_sse(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)
    input_items = resume_outputs(first_response, ["approve"])
    input_items[0]["call_id"] = interrupt_tool_call_id("fabricated")

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            previous_response_id=first_response.id,
            input=input_items,
            stream=True,
        )

    assert exc_info.value.status_code == HTTPStatus.CONFLICT
    assert not exc_info.value.response.headers["content-type"].startswith(
        "text/event-stream"
    )


async def test_repeated_resume_does_not_execute_completed_run_again(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_response = await create_response(openai_client)
    await resume_response(openai_client, first_response, "approve")

    graph_config = fastapi_app.state.graph_registry.get_graph(MODEL)
    graph = await graph_config.resolve_graph()

    def fail_execution(*_args, **_kwargs):
        msg = "a repeated resume must not execute the graph"
        raise AssertionError(msg)

    with monkeypatch.context() as retry_patch:
        retry_patch.setattr(graph, "astream", fail_execution)
        with pytest.raises(ConflictError):
            await resume_response(openai_client, first_response, "approve")


async def test_terminal_run_deletes_its_checkpoint_lineage(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    first_response = await create_response(openai_client)
    first_call = interrupt_calls(first_response)[0]
    assert_interrupt_arguments(first_call)
    await resume_response(openai_client, first_response, "approve")

    await assert_no_checkpoints(sqlite_checkpointer)
