import json
import uuid
from http import HTTPStatus

import pytest
from anyio import create_task_group, fail_after
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, BadRequestError, ConflictError, InternalServerError
from openai.types.responses import Response, ResponseFunctionToolCall

from tests.graph.support.interrupt import DEFAULT_INTERRUPT_PAYLOAD

from .support import (
    CONCURRENT_MODEL,
    INVALID_PAYLOAD_MODEL,
    MODEL,
    PARALLEL_MODEL,
    SEQUENTIAL_MODEL,
    assert_checkpoint_deleted,
)

EXPECTED_PARALLEL_INTERRUPTS = 2


async def _create_response(
    openai_client: AsyncOpenAI,
    *,
    model: str = MODEL,
    run_id: str | None = None,
) -> Response:
    metadata = {"langgraph_run_id": run_id} if run_id is not None else None
    return await openai_client.responses.create(
        model=model,
        input="Hi",
        metadata=metadata,
    )


def _interrupt_calls(response: Response) -> list[ResponseFunctionToolCall]:
    calls = [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]
    assert len(calls) == len(response.output)
    return calls


def _interrupt_arguments(call: ResponseFunctionToolCall) -> dict:
    assert call.name == "langgraph_interrupt"
    assert call.id is not None
    assert call.id.startswith("fc_")
    assert call.call_id.startswith("lg_interrupt_")
    arguments = json.loads(call.arguments)
    assert set(arguments) == {"run_id", "state_token", "payload"}
    assert uuid.UUID(arguments["run_id"]).int != 0
    assert arguments["state_token"]
    return arguments


def _resume_input(response: Response, values: list[object]) -> list[dict]:
    calls = _interrupt_calls(response)
    assert len(calls) == len(values)
    return [
        *[call.model_dump(mode="json", exclude_none=True) for call in calls],
        *[
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(
                    {"resume": value},
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            }
            for call, value in zip(calls, values, strict=True)
        ],
    ]


async def _resume_response(
    openai_client: AsyncOpenAI,
    response: Response,
    *values: object,
    model: str = MODEL,
) -> Response:
    return await openai_client.responses.create(
        model=model,
        input=_resume_input(response, list(values)),
    )


async def test_non_streaming_interrupt_uses_function_calls_and_resumes(
    openai_client: AsyncOpenAI,
) -> None:
    first = await _create_response(openai_client)

    calls = _interrupt_calls(first)
    assert len(calls) == 1
    arguments = _interrupt_arguments(calls[0])
    assert arguments["payload"] == DEFAULT_INTERRUPT_PAYLOAD

    final = await _resume_response(openai_client, first, "approve")
    assert final.output_text == "resumed:approve"


async def test_sdk_output_items_replay_without_custom_serialization(
    openai_client: AsyncOpenAI,
) -> None:
    first = await _create_response(openai_client)
    call = _interrupt_calls(first)[0]

    final = await openai_client.responses.create(
        model=MODEL,
        input=[
            *first.output,
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps({"resume": "approve"}),
            },
        ],
    )

    assert final.output_text == "resumed:approve"


async def test_interrupt_run_id_errors_keep_responses_params(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as invalid_exc:
        await _create_response(openai_client, run_id="not-a-uuid")
    assert invalid_exc.value.body["param"] == "metadata.langgraph_run_id"

    first = await _create_response(openai_client)
    with pytest.raises(BadRequestError) as mismatch_exc:
        await openai_client.responses.create(
            model=MODEL,
            input=_resume_input(first, ["approve"]),
            metadata={"langgraph_run_id": str(uuid.uuid4())},
        )
    assert mismatch_exc.value.body["param"] == "input"


async def test_parallel_interrupt_requires_and_resumes_complete_batch(
    openai_client: AsyncOpenAI,
) -> None:
    first = await _create_response(openai_client, model=PARALLEL_MODEL)
    calls = _interrupt_calls(first)
    assert len(calls) == EXPECTED_PARALLEL_INTERRUPTS
    arguments = [_interrupt_arguments(call) for call in calls]
    assert {item["payload"]["question"] for item in arguments} == {"left", "right"}
    assert len({item["run_id"] for item in arguments}) == 1
    assert len({item["state_token"] for item in arguments}) == 1

    incomplete = _resume_input(first, ["first", "second"])
    incomplete.pop()
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model=PARALLEL_MODEL,
            input=incomplete,
        )
    assert exc_info.value.body["param"] == "input"
    assert "Every interrupt" in exc_info.value.body["message"]

    final = await _resume_response(
        openai_client,
        first,
        "first",
        "second",
        model=PARALLEL_MODEL,
    )
    assert final.output_text == "first,second"


async def test_malformed_interrupt_arguments_return_input_error(
    openai_client: AsyncOpenAI,
) -> None:
    first = await _create_response(openai_client)
    resume_input = _resume_input(first, ["approve"])
    resume_input[0]["arguments"] = "{"

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(model=MODEL, input=resume_input)

    assert exc_info.value.body["param"] == "input"
    assert "must be valid JSON" in exc_info.value.body["message"]


async def test_interrupt_stream_uses_function_argument_lifecycle(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await openai_client.responses.create(
        model=MODEL,
        input="Hi",
        stream=True,
    )
    events = [event async for event in stream]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    added = events[2]
    delta = events[3]
    arguments_done = events[4]
    item_done = events[5]
    assert added.output_index == delta.output_index == arguments_done.output_index == 0
    assert added.item.id == delta.item_id == arguments_done.item_id == item_done.item.id
    assert added.item.call_id == item_done.item.call_id
    assert not added.item.arguments
    assert delta.delta == arguments_done.arguments == item_done.item.arguments
    assert _interrupt_arguments(item_done.item)["payload"] == DEFAULT_INTERRUPT_PAYLOAD


async def test_retry_reemits_pending_interrupt_without_execution(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid.uuid4()).upper()
    first = await _create_response(openai_client, run_id=run_id)
    graph = await fastapi_app.state.graph_registry.get_graph(MODEL).resolve_graph()

    def fail_execution(*_args, **_kwargs):
        msg = "a pending retry must not execute the graph"
        raise AssertionError(msg)

    with monkeypatch.context() as retry_patch:
        retry_patch.setattr(graph, "ainvoke", fail_execution)
        retried = await _create_response(openai_client, run_id=run_id)

    first_calls = _interrupt_calls(first)
    retried_calls = _interrupt_calls(retried)
    assert [call.call_id for call in retried_calls] == [
        call.call_id for call in first_calls
    ]
    assert [call.arguments for call in retried_calls] == [
        call.arguments for call in first_calls
    ]
    assert [call.id for call in retried_calls] != [call.id for call in first_calls]


async def test_concurrent_resume_returns_run_busy_and_executes_once(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    first = await _create_response(openai_client, model=CONCURRENT_MODEL)
    resume_input = _resume_input(first, ["approve"])
    responses: list[Response] = []

    async def complete_first_resume() -> None:
        response = await openai_client.responses.create(
            model=CONCURRENT_MODEL,
            input=resume_input,
        )
        responses.append(response)

    async with create_task_group() as task_group:
        task_group.start_soon(complete_first_resume)
        with fail_after(1):
            await fastapi_app.state.resume_entered.wait()
        try:
            with fail_after(1), pytest.raises(ConflictError) as exc_info:
                await openai_client.with_options(max_retries=0).responses.create(
                    model=CONCURRENT_MODEL,
                    input=resume_input,
                )
        finally:
            fastapi_app.state.resume_release.set()

    assert len(responses) == 1
    assert responses[0].output_text == "approve"
    assert fastapi_app.state.side_effects == {"count": 1}
    assert exc_info.value.body["code"] == "run_busy"


async def test_stale_sequential_resume_returns_conflict_before_sse(
    openai_client: AsyncOpenAI,
) -> None:
    first = await _create_response(openai_client, model=SEQUENTIAL_MODEL)
    first_input = _resume_input(first, ["one"])
    second = await openai_client.responses.create(
        model=SEQUENTIAL_MODEL,
        input=first_input,
    )

    first_call = _interrupt_calls(first)[0]
    second_call = _interrupt_calls(second)[0]
    assert first_call.call_id == second_call.call_id
    assert (
        _interrupt_arguments(first_call)["state_token"]
        != _interrupt_arguments(second_call)["state_token"]
    )

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.responses.create(
            model=SEQUENTIAL_MODEL,
            input=first_input,
            stream=True,
        )
    assert exc_info.value.status_code == HTTPStatus.CONFLICT
    assert exc_info.value.body["param"] == "input"
    assert exc_info.value.body["code"] == "interrupt_state_conflict"
    assert not exc_info.value.response.headers["content-type"].startswith(
        "text/event-stream"
    )

    final = await _resume_response(
        openai_client,
        second,
        "two",
        model=SEQUENTIAL_MODEL,
    )
    assert final.output_text == "one,two"


async def test_invalid_interrupt_payload_is_server_error_and_cleans_checkpoint(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    run_id = str(uuid.uuid4())
    with pytest.raises(InternalServerError) as exc_info:
        await _create_response(
            openai_client,
            model=INVALID_PAYLOAD_MODEL,
            run_id=run_id,
        )

    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert exc_info.value.body["type"] == "server_error"
    assert exc_info.value.body["message"] == (
        "LangGraph interrupt payloads must be valid JSON values."
    )
    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
    )


async def test_streaming_invalid_interrupt_payload_fails_and_cleans_checkpoint(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    run_id = str(uuid.uuid4())
    stream = await openai_client.responses.create(
        model=INVALID_PAYLOAD_MODEL,
        input="Hi",
        metadata={"langgraph_run_id": run_id},
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


async def test_terminal_response_deletes_checkpoint_lineage(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    first = await _create_response(openai_client)
    arguments = _interrupt_arguments(_interrupt_calls(first)[0])

    await _resume_response(openai_client, first, "approve")

    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=MODEL,
        run_id=arguments["run_id"],
    )
