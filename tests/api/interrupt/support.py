import json
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI

from langgraph_openai_serve.graph.interrupt.state import checkpoint_key

MODEL = "interruptible"
PARALLEL_MODEL = "parallel-interrupts"
SEQUENTIAL_MODEL = "sequential-interrupts"
CONCURRENT_MODEL = "concurrent-resume"
INVALID_PAYLOAD_MODEL = "invalid-interrupt-payload"
NESTED_MODEL = "nested-parallel-interrupts"
NESTED_SEQUENTIAL_MODEL = "nested-sequential-interrupts"
CHECKPOINT_SCOPE_HEADER = "x-test-checkpoint-scope"


async def create_completion(
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
        extra_headers=_checkpoint_scope_headers(checkpoint_scope),
    )


def assert_interrupt_arguments(tool_call) -> dict:
    assert tool_call.function is not None
    assert tool_call.function.name == "langgraph_interrupt"
    assert tool_call.id is not None
    assert tool_call.id.startswith("lg_interrupt_")
    arguments = json.loads(tool_call.function.arguments)
    assert set(arguments) == {"run_id", "state_token", "payload"}
    assert uuid.UUID(arguments["run_id"]).int != 0
    assert arguments["state_token"]
    return arguments


def resume_messages(response, values: list[object]) -> list[dict]:
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


async def resume_interrupt(
    openai_client: AsyncOpenAI,
    response,
    *resume_values: object,
    model: str = MODEL,
    checkpoint_scope: str | None = None,
):
    return await openai_client.chat.completions.create(
        model=model,
        messages=resume_messages(response, list(resume_values)),
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
