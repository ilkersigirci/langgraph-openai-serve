"""Test support for OpenAI interrupt testing."""

import json

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseFunctionToolCall

from langgraph_openai_serve.graph.interrupt.state import checkpoint_key

MODEL = "interruptible"
PARALLEL_MODEL = "parallel-interrupts"
SEQUENTIAL_MODEL = "sequential-interrupts"
CONCURRENT_MODEL = "concurrent-resume"
INVALID_PAYLOAD_MODEL = "invalid-interrupt-payload"
NESTED_MODEL = "nested-parallel-interrupts"
NESTED_SEQUENTIAL_MODEL = "nested-sequential-interrupts"
CHECKPOINT_SCOPE_HEADER = "x-test-checkpoint-scope"


async def create_response(
    openai_client: AsyncOpenAI,
    *,
    model: str = MODEL,
    stream: bool = False,
    run_id: str | None = None,
    checkpoint_scope: str | None = None,
) -> Response:
    metadata = {"langgraph_run_id": run_id} if run_id is not None else None
    return await openai_client.responses.create(
        model=model,
        input="Hi",
        stream=stream,
        metadata=metadata,
        extra_headers=_checkpoint_scope_headers(checkpoint_scope),
    )


def interrupt_calls(response: Response) -> list[ResponseFunctionToolCall]:
    calls = [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]
    assert len(calls) == len(response.output)
    return calls


def assert_interrupt_arguments(call: ResponseFunctionToolCall) -> dict:
    assert call.name == "langgraph_interrupt"
    assert call.call_id.startswith("call_lg_")
    arguments = json.loads(call.arguments)
    assert isinstance(arguments, dict)
    return arguments


def resume_outputs(response: Response, values: list[object]) -> list[dict[str, str]]:
    calls = interrupt_calls(response)
    assert len(calls) == len(values)
    return [
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": value if isinstance(value, str) else json.dumps(value),
        }
        for call, value in zip(calls, values, strict=True)
    ]


async def resume_response(
    openai_client: AsyncOpenAI,
    response: Response,
    *resume_values: object,
    model: str = MODEL,
    checkpoint_scope: str | None = None,
) -> Response:
    return await openai_client.responses.create(
        model=model,
        previous_response_id=response.id,
        input=resume_outputs(response, list(resume_values)),
        extra_headers=_checkpoint_scope_headers(checkpoint_scope),
    )


def _checkpoint_scope_headers(scope: str | None) -> dict[str, str] | None:
    return {CHECKPOINT_SCOPE_HEADER: scope} if scope is not None else None


async def assert_checkpoint_deleted(
    checkpointer: AsyncSqliteSaver,
    *,
    model: str,
    run_id: str,
) -> None:
    checkpoint = await checkpointer.aget_tuple(
        {
            "configurable": {
                "thread_id": checkpoint_key(model, run_id),
            }
        }
    )

    assert checkpoint is None
