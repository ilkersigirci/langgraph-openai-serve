import asyncio
import json
import uuid
from http import HTTPStatus
from typing import Any

import pytest
from anyio import fail_after
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import (
    APIError,
    AsyncOpenAI,
    BadRequestError,
    ConflictError,
    InternalServerError,
)

from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from langgraph_openai_serve.graph.utils import checkpoint_key
from tests.graph.support.interrupt import (
    InterruptAnswerState,
    make_interrupt_graph,
    make_parallel_interrupt_graph,
    make_parallel_nested_interrupt_graph,
    make_sequential_interrupt_graph,
)

MODEL = "interruptible"
PARALLEL_MODEL = "parallel-interrupts"
SEQUENTIAL_MODEL = "sequential-interrupts"
CONCURRENT_MODEL = "concurrent-resume"
INVALID_PAYLOAD_MODEL = "invalid-interrupt-payload"
NESTED_MODEL = "nested-parallel-interrupts"
INTERRUPT_PAYLOAD = {"question": "Approve?"}
EXPECTED_PARALLEL_INTERRUPTS = 2


async def _create_completion(
    openai_client: AsyncOpenAI,
    *,
    model: str = MODEL,
    stream: bool = False,
    run_id: str | None = None,
    checkpoint_scope: str | None = None,
):
    metadata = {"langgraph_run_id": run_id} if run_id is not None else None
    return await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        stream=stream,
        metadata=metadata,
        extra_headers=(
            {"x-test-checkpoint-scope": checkpoint_scope}
            if checkpoint_scope is not None
            else None
        ),
    )


def _interrupt_arguments(tool_call) -> dict:
    assert tool_call.function is not None
    assert tool_call.function.name == "langgraph_interrupt"
    assert tool_call.id is not None
    assert tool_call.id.startswith("lg_interrupt_")
    arguments = json.loads(tool_call.function.arguments)
    assert set(arguments) == {"run_id", "state_token", "payload"}
    assert uuid.UUID(arguments["run_id"])
    assert arguments["state_token"]
    return arguments


def _resume_messages(response, values: list[object]) -> list[dict]:
    assistant = response.choices[0].message
    tool_calls = assistant.tool_calls or []
    assert len(tool_calls) == len(values)
    return [
        assistant.model_dump(mode="json", exclude_none=True),
        *[
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"resume": value}),
            }
            for tool_call, value in zip(tool_calls, values, strict=True)
        ],
    ]


async def _resume_interrupt(
    openai_client: AsyncOpenAI,
    response,
    *resume_values: object,
    model: str = MODEL,
    checkpoint_scope: str | None = None,
):
    return await openai_client.chat.completions.create(
        model=model,
        messages=_resume_messages(response, list(resume_values)),
        extra_headers=(
            {"x-test-checkpoint-scope": checkpoint_scope}
            if checkpoint_scope is not None
            else None
        ),
    )


@pytest.fixture
def fastapi_app(sqlite_checkpointer: AsyncSqliteSaver) -> FastAPI:
    coordinator = InMemoryRunCoordinator()
    resume_entered = asyncio.Event()
    resume_release = asyncio.Event()
    side_effects = {"count": 0}

    async def concurrent_approval(
        _state: InterruptAnswerState,
    ) -> dict[str, list[str]]:
        decision = interrupt({"question": "concurrent"})
        side_effects["count"] += 1
        resume_entered.set()
        await resume_release.wait()
        return {"answers": [str(decision)]}

    concurrent_graph = (
        StateGraph(InterruptAnswerState)
        .add_node("approve", concurrent_approval)
        .add_edge(START, "approve")
        .add_edge("approve", END)
        .compile(checkpointer=sqlite_checkpointer)
    )

    def interrupt_config(graph: Any, **kwargs: Any) -> GraphConfig:
        return GraphConfig(
            graph=graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            run_coordinator=coordinator,
            **kwargs,
        )

    graph_registry = GraphRegistry(
        registry={
            MODEL: interrupt_config(
                make_interrupt_graph(
                    INTERRUPT_PAYLOAD,
                    checkpointer=sqlite_checkpointer,
                ),
            ),
            PARALLEL_MODEL: interrupt_config(
                make_parallel_interrupt_graph(sqlite_checkpointer),
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_text=lambda output: ",".join(sorted(output["answers"])),
            ),
            SEQUENTIAL_MODEL: interrupt_config(
                make_sequential_interrupt_graph(sqlite_checkpointer),
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_text=lambda output: ",".join(output["answers"]),
            ),
            CONCURRENT_MODEL: interrupt_config(
                concurrent_graph,
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_text=lambda output: output["answers"][0],
            ),
            INVALID_PAYLOAD_MODEL: interrupt_config(
                make_interrupt_graph(
                    {"value": float("nan")},
                    checkpointer=sqlite_checkpointer,
                ),
            ),
            NESTED_MODEL: interrupt_config(
                make_parallel_nested_interrupt_graph(sqlite_checkpointer),
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_text=lambda output: ",".join(sorted(output["answers"])),
            ),
        }
    )
    app = (
        LanggraphOpenaiServe(
            graphs=graph_registry,
            checkpoint_scope=lambda request: request.headers.get(
                "x-test-checkpoint-scope",
                "default",
            ),
        )
        .bind_openai_api()
        .app
    )
    app.state.test_checkpointer = sqlite_checkpointer
    app.state.resume_entered = resume_entered
    app.state.resume_release = resume_release
    app.state.side_effects = side_effects
    return app


async def test_non_streaming_interrupt_matches_contract_and_resumes(
    openai_client: AsyncOpenAI,
) -> None:
    response = await _create_completion(openai_client)

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    arguments = _interrupt_arguments(choice.message.tool_calls[0])
    assert arguments["payload"] == INTERRUPT_PAYLOAD
    final_response = await _resume_interrupt(openai_client, response, "approve")

    assert final_response.choices[0].message.content == "resumed:approve"


async def test_id_mapped_json_null_is_a_valid_resume_value(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await _create_completion(openai_client)
    final_response = await _resume_interrupt(openai_client, first_response, None)

    assert final_response.choices[0].message.content == "resumed:None"


async def test_same_run_id_reemits_pending_batch_without_executing_graph(
    openai_client: AsyncOpenAI,
) -> None:
    run_id = str(uuid.uuid4()).upper()
    first_response = await _create_completion(openai_client, run_id=run_id)
    recovered_response = await _create_completion(openai_client, run_id=run_id)

    first_calls = first_response.choices[0].message.tool_calls or []
    recovered_calls = recovered_response.choices[0].message.tool_calls or []
    assert _interrupt_arguments(first_calls[0])["run_id"] == run_id.lower()
    assert [call.model_dump(mode="json") for call in recovered_calls] == [
        call.model_dump(mode="json") for call in first_calls
    ]

    final_response = await _resume_interrupt(
        openai_client,
        recovered_response,
        "approve",
    )
    assert final_response.choices[0].message.content == "resumed:approve"


async def test_same_run_id_is_isolated_by_server_checkpoint_scope(
    openai_client: AsyncOpenAI,
) -> None:
    run_id = str(uuid.uuid4())
    tenant_a = await _create_completion(
        openai_client,
        run_id=run_id,
        checkpoint_scope="tenant-a",
    )
    tenant_b = await _create_completion(
        openai_client,
        run_id=run_id,
        checkpoint_scope="tenant-b",
    )

    with pytest.raises(ConflictError):
        await _resume_interrupt(
            openai_client,
            tenant_a,
            "approve",
            checkpoint_scope="tenant-b",
        )

    response_a = await _resume_interrupt(
        openai_client,
        tenant_a,
        "approve",
        checkpoint_scope="tenant-a",
    )
    response_b = await _resume_interrupt(
        openai_client,
        tenant_b,
        "reject",
        checkpoint_scope="tenant-b",
    )
    assert response_a.choices[0].message.content == "resumed:approve"
    assert response_b.choices[0].message.content == "resumed:reject"


async def test_invalid_caller_run_id_returns_400(openai_client: AsyncOpenAI) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await _create_completion(openai_client, run_id="shared-chat")

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.body["param"] == "metadata.langgraph_run_id"


async def test_streaming_interrupt_matches_openai_tool_call_contract(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await _create_completion(openai_client, stream=True)
    chunks = [chunk async for chunk in stream]

    tool_call_chunks = [
        chunk for chunk in chunks if chunk.choices[0].delta.tool_calls is not None
    ]
    assert len(tool_call_chunks) == 1
    tool_call = tool_call_chunks[0].choices[0].delta.tool_calls[0]
    assert tool_call.index == 0
    _interrupt_arguments(tool_call)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


async def test_invalid_interrupt_payload_returns_openai_server_error(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    run_id = str(uuid.uuid4())
    with pytest.raises(InternalServerError) as exc_info:
        await _create_completion(
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
    config = {
        "configurable": {
            "thread_id": checkpoint_key(INVALID_PAYLOAD_MODEL, run_id),
        }
    }
    assert await fastapi_app.state.test_checkpointer.aget_tuple(config) is None


async def test_streaming_invalid_interrupt_payload_deletes_checkpoint(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    run_id = str(uuid.uuid4())
    stream = await _create_completion(
        openai_client,
        model=INVALID_PAYLOAD_MODEL,
        run_id=run_id,
        stream=True,
    )

    with pytest.raises(APIError, match="Internal server error"):
        _ = [chunk async for chunk in stream]

    config = {
        "configurable": {
            "thread_id": checkpoint_key(INVALID_PAYLOAD_MODEL, run_id),
        }
    }
    assert await fastapi_app.state.test_checkpointer.aget_tuple(config) is None


async def test_parallel_interrupts_are_one_tool_call_batch_and_resume_by_id(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await _create_completion(openai_client, model=PARALLEL_MODEL)
    tool_calls = first_response.choices[0].message.tool_calls or []

    assert len(tool_calls) == EXPECTED_PARALLEL_INTERRUPTS
    arguments = [_interrupt_arguments(tool_call) for tool_call in tool_calls]
    assert {item["payload"]["question"] for item in arguments} == {"left", "right"}
    assert len({item["state_token"] for item in arguments}) == 1
    assert len({item["run_id"] for item in arguments}) == 1

    final_response = await _resume_interrupt(
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
    first_response = await _create_completion(openai_client, model=NESTED_MODEL)
    tool_calls = first_response.choices[0].message.tool_calls or []

    assert len(tool_calls) == EXPECTED_PARALLEL_INTERRUPTS
    assert {
        _interrupt_arguments(tool_call)["payload"]["question"]
        for tool_call in tool_calls
    } == {"nested-a", "nested-b"}
    values = [
        "first"
        if _interrupt_arguments(tool_call)["payload"]["question"] == "nested-a"
        else "second"
        for tool_call in tool_calls
    ]

    final_response = await _resume_interrupt(
        openai_client,
        first_response,
        *values,
        model=NESTED_MODEL,
    )

    assert final_response.choices[0].message.content == (
        "nested-a:first,nested-b:second"
    )


async def test_fabricated_interrupt_id_cannot_resume_pending_state(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await _create_completion(openai_client)
    messages = _resume_messages(first_response, ["approve"])
    assistant_call = messages[0]["tool_calls"][0]
    assistant_call["id"] = "lg_interrupt_fabricated"
    messages[1]["tool_call_id"] = "lg_interrupt_fabricated"

    with pytest.raises(ConflictError) as exc_info:
        await openai_client.chat.completions.create(model=MODEL, messages=messages)

    assert exc_info.value.status_code == HTTPStatus.CONFLICT


async def test_modified_replayed_payload_does_not_change_resume_target(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await _create_completion(openai_client)
    messages = _resume_messages(first_response, ["approve"])
    assistant_call = messages[0]["tool_calls"][0]
    arguments = json.loads(assistant_call["function"]["arguments"])
    arguments["payload"] = {"question": "Approve a different action?"}
    assistant_call["function"]["arguments"] = json.dumps(arguments)

    response = await openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assert response.choices[0].message.content == "resumed:approve"


async def test_checkpoint_token_disambiguates_sequential_reused_interrupt_id(
    openai_client: AsyncOpenAI,
) -> None:
    first_pause = await _create_completion(openai_client, model=SEQUENTIAL_MODEL)
    first_messages = _resume_messages(first_pause, ["one"])
    second_pause = await openai_client.chat.completions.create(
        model=SEQUENTIAL_MODEL,
        messages=first_messages,
    )

    first_call = first_pause.choices[0].message.tool_calls[0]
    second_call = second_pause.choices[0].message.tool_calls[0]
    assert first_call.id == second_call.id
    first_arguments = _interrupt_arguments(first_call)
    second_arguments = _interrupt_arguments(second_call)
    assert first_arguments["state_token"] != second_arguments["state_token"]

    with pytest.raises(ConflictError):
        await openai_client.chat.completions.create(
            model=SEQUENTIAL_MODEL,
            messages=first_messages,
        )

    final_response = await _resume_interrupt(
        openai_client,
        second_pause,
        "two",
        model=SEQUENTIAL_MODEL,
    )
    assert final_response.choices[0].message.content == "one,two"


async def test_streaming_state_conflict_returns_409_before_sse(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await _create_completion(openai_client)
    messages = _resume_messages(first_response, ["approve"])
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
) -> None:
    first_response = await _create_completion(openai_client)
    messages = _resume_messages(first_response, ["approve"])
    await openai_client.chat.completions.create(model=MODEL, messages=messages)

    with pytest.raises(ConflictError):
        await openai_client.chat.completions.create(model=MODEL, messages=messages)


async def test_concurrent_resume_executes_post_interrupt_work_once(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    first_response = await _create_completion(openai_client, model=CONCURRENT_MODEL)
    messages = _resume_messages(first_response, ["approve"])

    first_resume = asyncio.create_task(
        openai_client.chat.completions.create(
            model=CONCURRENT_MODEL,
            messages=messages,
        )
    )
    with fail_after(1):
        await fastapi_app.state.resume_entered.wait()
    competing_resume = asyncio.create_task(
        openai_client.with_options(max_retries=0).chat.completions.create(
            model=CONCURRENT_MODEL,
            messages=messages,
        )
    )

    try:
        with pytest.raises(ConflictError) as exc_info:
            await competing_resume
    finally:
        fastapi_app.state.resume_release.set()
    response = await first_resume

    assert response.choices[0].message.content == "approve"
    assert fastapi_app.state.side_effects == {"count": 1}
    assert exc_info.value.body["code"] == "run_busy"


async def test_terminal_run_deletes_its_checkpoint_lineage(
    openai_client: AsyncOpenAI,
    fastapi_app: FastAPI,
) -> None:
    first_response = await _create_completion(openai_client)
    arguments = _interrupt_arguments(first_response.choices[0].message.tool_calls[0])
    await _resume_interrupt(openai_client, first_response, "approve")

    snapshot = await fastapi_app.state.test_checkpointer.aget_tuple(
        {
            "configurable": {
                "thread_id": checkpoint_key(MODEL, arguments["run_id"]),
            }
        }
    )
    assert snapshot is None
