import json
import uuid
from http import HTTPStatus

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, BadRequestError, ConflictError

from .support import (
    MODEL,
    NESTED_SEQUENTIAL_MODEL,
    SEQUENTIAL_MODEL,
    assert_checkpoint_deleted,
    assert_interrupt_arguments,
    create_completion,
    resume_interrupt,
    resume_messages,
)


async def test_retry_with_same_run_id_reemits_pending_batch_without_execution(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid.uuid4()).upper()
    first_response = await create_completion(openai_client, run_id=run_id)
    graph_config = fastapi_app.state.graph_registry.get_graph(MODEL)
    graph = await graph_config.resolve_graph()

    def fail_execution(*_args, **_kwargs):
        raise AssertionError("a pending retry must not execute the graph")

    with monkeypatch.context() as retry_patch:
        retry_patch.setattr(graph, "astream", fail_execution)
        recovered_response = await create_completion(openai_client, run_id=run_id)

    first_calls = first_response.choices[0].message.tool_calls or []
    recovered_calls = recovered_response.choices[0].message.tool_calls or []
    assert assert_interrupt_arguments(first_calls[0])["run_id"] == run_id.lower()
    assert [call.model_dump(mode="json") for call in recovered_calls] == [
        call.model_dump(mode="json") for call in first_calls
    ]

    final_response = await resume_interrupt(
        openai_client,
        recovered_response,
        "approve",
    )
    assert final_response.choices[0].message.content == "resumed:approve"


async def test_same_run_id_is_isolated_by_server_checkpoint_scope(
    openai_client: AsyncOpenAI,
) -> None:
    run_id = str(uuid.uuid4())
    tenant_a = await create_completion(
        openai_client,
        run_id=run_id,
        checkpoint_scope="tenant-a",
    )
    tenant_b = await create_completion(
        openai_client,
        run_id=run_id,
        checkpoint_scope="tenant-b",
    )

    with pytest.raises(ConflictError):
        await resume_interrupt(
            openai_client,
            tenant_a,
            "approve",
            checkpoint_scope="tenant-b",
        )

    response_a = await resume_interrupt(
        openai_client,
        tenant_a,
        "approve",
        checkpoint_scope="tenant-a",
    )
    response_b = await resume_interrupt(
        openai_client,
        tenant_b,
        "reject",
        checkpoint_scope="tenant-b",
    )
    assert response_a.choices[0].message.content == "resumed:approve"
    assert response_b.choices[0].message.content == "resumed:reject"


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
        await create_completion(openai_client, run_id=run_id)

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "metadata.langgraph_run_id"


async def test_resume_rejects_mismatched_caller_run_id(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client)
    messages = resume_messages(first_response, ["approve"])

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            metadata={"langgraph_run_id": str(uuid.uuid4())},
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "messages"


async def test_fabricated_interrupt_id_cannot_resume_pending_state(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client)
    messages = resume_messages(first_response, ["approve"])
    assistant_call = messages[0]["tool_calls"][0]
    assistant_call["id"] = "lg_interrupt_fabricated"
    messages[1]["tool_call_id"] = "lg_interrupt_fabricated"

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.chat.completions.create(model=MODEL, messages=messages)

    assert exc_info.value.status_code == HTTPStatus.CONFLICT


async def test_modified_replayed_payload_does_not_change_resume_target(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client)
    messages = resume_messages(first_response, ["approve"])
    assistant_call = messages[0]["tool_calls"][0]
    arguments = json.loads(assistant_call["function"]["arguments"])
    arguments["payload"] = {"question": "Approve a different action?"}
    assistant_call["function"]["arguments"] = json.dumps(arguments)

    response = await openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assert response.choices[0].message.content == "resumed:approve"


@pytest.mark.parametrize("model", [SEQUENTIAL_MODEL, NESTED_SEQUENTIAL_MODEL])
async def test_checkpoint_token_disambiguates_sequential_reused_interrupt_id(
    openai_client: AsyncOpenAI,
    model: str,
) -> None:
    first_pause = await create_completion(openai_client, model=model)
    first_messages = resume_messages(first_pause, ["one"])
    second_pause = await openai_client.chat.completions.create(
        model=model,
        messages=first_messages,
    )

    first_call = first_pause.choices[0].message.tool_calls[0]
    second_call = second_pause.choices[0].message.tool_calls[0]
    assert first_call.id == second_call.id
    first_arguments = assert_interrupt_arguments(first_call)
    second_arguments = assert_interrupt_arguments(second_call)
    assert first_arguments["state_token"] != second_arguments["state_token"]

    with pytest.raises(ConflictError):
        await openai_client.chat.completions.create(
            model=model,
            messages=first_messages,
        )

    final_response = await resume_interrupt(
        openai_client,
        second_pause,
        "two",
        model=model,
    )
    assert final_response.choices[0].message.content == "one,two"


async def test_streaming_state_conflict_returns_409_before_sse(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await create_completion(openai_client)
    messages = resume_messages(first_response, ["approve"])
    assistant_call = messages[0]["tool_calls"][0]
    arguments = json.loads(assistant_call["function"]["arguments"])
    arguments["state_token"] = "stale-state-token"
    assistant_call["function"]["arguments"] = json.dumps(arguments)

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
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
    first_response = await create_completion(openai_client)
    messages = resume_messages(first_response, ["approve"])
    await openai_client.chat.completions.create(model=MODEL, messages=messages)

    graph_config = fastapi_app.state.graph_registry.get_graph(MODEL)
    graph = await graph_config.resolve_graph()

    def fail_execution(*_args, **_kwargs):
        raise AssertionError("a repeated resume must not execute the graph")

    with monkeypatch.context() as retry_patch:
        retry_patch.setattr(graph, "astream", fail_execution)
        with pytest.raises(ConflictError):
            await openai_client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )


async def test_terminal_run_deletes_its_checkpoint_lineage(
    openai_client: AsyncOpenAI,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    first_response = await create_completion(openai_client)
    arguments = assert_interrupt_arguments(
        first_response.choices[0].message.tool_calls[0]
    )
    await resume_interrupt(openai_client, first_response, "approve")

    await assert_checkpoint_deleted(
        sqlite_checkpointer,
        model=MODEL,
        run_id=arguments["run_id"],
    )
