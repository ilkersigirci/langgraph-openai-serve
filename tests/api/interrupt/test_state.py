import uuid
from http import HTTPStatus

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, BadRequestError, ConflictError

from langgraph_openai_serve.api.responses.interrupts import interrupt_tool_call_id

from .support import (
    MODEL,
    NESTED_SEQUENTIAL_MODEL,
    SEQUENTIAL_MODEL,
    assert_checkpoint_deleted,
    assert_interrupt_arguments,
    create_response,
    interrupt_calls,
    resume_outputs,
    resume_response,
)


async def test_retry_with_same_run_id_reemits_pending_batch_without_execution(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid.uuid4()).upper()
    first_response = await create_response(openai_client, run_id=run_id)
    graph_config = fastapi_app.state.graph_registry.get_graph(MODEL)
    graph = await graph_config.resolve_graph()

    def fail_execution(*_args, **_kwargs):
        msg = "a pending retry must not execute the graph"
        raise AssertionError(msg)

    with monkeypatch.context() as retry_patch:
        retry_patch.setattr(graph, "astream", fail_execution)
        recovered_response = await create_response(openai_client, run_id=run_id)

    first_calls = interrupt_calls(first_response)
    recovered_calls = interrupt_calls(recovered_response)
    clean_id = run_id.lower().replace("-", "")
    assert first_response.id.startswith(f"resp_lg_{clean_id}_")
    assert recovered_response.id.startswith(f"resp_lg_{clean_id}_")
    assert recovered_response.id != first_response.id
    assert [call.arguments for call in recovered_calls] == [
        call.arguments for call in first_calls
    ]

    final_response = await resume_response(
        openai_client,
        recovered_response,
        "approve",
    )
    assert final_response.output_text == "resumed:approve"


async def test_same_run_id_is_isolated_by_server_checkpoint_scope(
    openai_client: AsyncOpenAI,
) -> None:
    run_id = str(uuid.uuid4())
    tenant_a = await create_response(
        openai_client,
        run_id=run_id,
        checkpoint_scope="tenant-a",
    )
    tenant_b = await create_response(
        openai_client,
        run_id=run_id,
        checkpoint_scope="tenant-b",
    )

    with pytest.raises(ConflictError):
        await resume_response(
            openai_client,
            tenant_a,
            "approve",
            checkpoint_scope="tenant-b",
        )

    response_a = await resume_response(
        openai_client,
        tenant_a,
        "approve",
        checkpoint_scope="tenant-a",
    )
    response_b = await resume_response(
        openai_client,
        tenant_b,
        "reject",
        checkpoint_scope="tenant-b",
    )
    assert response_a.output_text == "resumed:approve"
    assert response_b.output_text == "resumed:reject"


@pytest.mark.parametrize(
    "run_id",
    [
        pytest.param("shared-chat", id="not-a-uuid"),
        pytest.param("00000000-0000-0000-0000-000000000000", id="nil-uuid"),
    ],
)
async def test_invalid_caller_run_id_returns_400(
    openai_client: AsyncOpenAI,
    run_id: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await create_response(openai_client, run_id=run_id)

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "metadata.lgos_run_id"


async def test_resume_rejects_mismatched_caller_run_id(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)
    input_items = resume_outputs(first_response, ["approve"])

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            previous_response_id=first_response.id,
            input=input_items,
            metadata={"lgos_run_id": str(uuid.uuid4())},
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "metadata.lgos_run_id"


async def test_fabricated_interrupt_id_cannot_resume_pending_state(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)
    call_id = interrupt_calls(first_response)[0].call_id
    state_token = call_id.removeprefix("call_lg_").partition("_")[0]
    input_items = resume_outputs(first_response, ["approve"])
    input_items[0]["call_id"] = interrupt_tool_call_id(
        "fabricated",
        state_token,
        response_id=first_response.id,
    )

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.responses.create(
            model=MODEL,
            previous_response_id=first_response.id,
            input=input_items,
        )

    assert exc_info.value.status_code == HTTPStatus.CONFLICT


@pytest.mark.parametrize("model", [SEQUENTIAL_MODEL, NESTED_SEQUENTIAL_MODEL])
async def test_checkpoint_token_disambiguates_sequential_reused_interrupt_id(
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


@pytest.mark.parametrize("stream", [False, True])
async def test_interrupt_outputs_must_belong_to_previous_response(
    openai_client: AsyncOpenAI,
    stream: bool,
) -> None:
    first_pause = await create_response(openai_client, model=SEQUENTIAL_MODEL)
    second_pause = await resume_response(
        openai_client, first_pause, "one", model=SEQUENTIAL_MODEL
    )

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model=SEQUENTIAL_MODEL,
            previous_response_id=first_pause.id,
            input=resume_outputs(second_pause, ["two"]),
            stream=stream,
        )

    assert exc_info.value.body["param"] == "previous_response_id"
    final = await resume_response(
        openai_client, second_pause, "two", model=SEQUENTIAL_MODEL
    )
    assert final.output_text == "one,two"


async def test_streaming_state_conflict_returns_409_before_sse(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_response(openai_client)
    input_items = resume_outputs(first_response, ["approve"])
    interrupt_id = input_items[0]["call_id"].removeprefix("call_lg_").split("_", 2)[2]
    input_items[0]["call_id"] = interrupt_tool_call_id(
        interrupt_id, "f" * 64, response_id=first_response.id
    )

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
    run_id = str(uuid.uuid4())
    first_response = await create_response(openai_client, run_id=run_id)
    first_call = interrupt_calls(first_response)[0]
    assert_interrupt_arguments(first_call)
    await resume_response(openai_client, first_response, "approve")

    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=MODEL,
        run_id=run_id,
    )
