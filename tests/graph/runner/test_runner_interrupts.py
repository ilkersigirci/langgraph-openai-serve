import asyncio
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from anyio import create_task_group, get_cancelled_exc_class, sleep_forever
from anyio.lowlevel import checkpoint
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph

from langgraph_openai_serve.api.chat.utils.responses import response_message
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.runner import (
    LangGraphInterruptBatch,
    invoke_run,
    run_langgraph,
    run_langgraph_stream,
)
from langgraph_openai_serve.graph.utils import (
    RUN_METADATA_KEY,
    GraphRun,
    checkpoint_key,
    prepare_run,
)
from tests.graph.support.interrupt import (
    DEFAULT_INTERRUPT_PAYLOAD,
    make_interrupt_graph,
    make_parallel_interrupt_graph,
)
from tests.graph.support.message import make_message_graph
from tests.graph.support.schemas import MessageState

EXPECTED_PARALLEL_INTERRUPTS = 2
SHA256_HEX_LENGTH = 64
RUN_ID = "11111111-1111-4111-8111-111111111111"
STREAM_RUN_ID = "22222222-2222-4222-8222-222222222222"


class AsyncReadOnlyCheckpointer(BaseCheckpointSaver):
    async def aget_tuple(self, config: RunnableConfig):
        return None


class AsyncCheckpointerWithoutPendingWrites(AsyncReadOnlyCheckpointer):
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return config

    async def adelete_thread(self, thread_id: str) -> None:
        return None


class AsyncCheckpointerWithoutDelete(AsyncCheckpointerWithoutPendingWrites):
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return None

    adelete_thread = BaseCheckpointSaver.adelete_thread


async def test_cancelled_preparation_finishes_lease_release(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    state_read_started = asyncio.Event()
    release_started = asyncio.Event()
    released = asyncio.Event()
    cancellation_propagated = asyncio.Event()

    graph = make_interrupt_graph(checkpointer=sqlite_checkpointer)

    async def blocked_state_read(*_args, **_kwargs):
        state_read_started.set()
        await sleep_forever()

    monkeypatch.setattr(graph, "aget_state", blocked_state_read)

    @asynccontextmanager
    async def coordinator(_key: str):
        try:
            yield
        finally:
            release_started.set()
            await checkpoint()
            released.set()

    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                run_coordinator=coordinator,
            )
        }
    )
    request = make_request("interruptible")

    async def run_preparation() -> None:
        try:
            await prepare_run(
                "interruptible",
                request.messages,
                registry,
                request,
            )
        except get_cancelled_exc_class():
            cancellation_propagated.set()

    async with create_task_group() as task_group:
        task_group.start_soon(run_preparation)
        await state_read_started.wait()
        task_group.cancel_scope.cancel()

    assert release_started.is_set()
    assert released.is_set()
    assert cancellation_propagated.is_set()


def test_checkpoint_key_is_model_scoped_and_does_not_expose_public_run_id() -> None:
    model_a_key = checkpoint_key("model-a", RUN_ID)
    model_b_key = checkpoint_key("model-b", RUN_ID)
    tenant_b_key = checkpoint_key("model-a", RUN_ID, scope="tenant-b")

    assert model_a_key != model_b_key
    assert model_a_key != tenant_b_key
    assert RUN_ID not in model_a_key
    assert len(model_a_key) == SHA256_HEX_LENGTH


async def test_thread_id_reaches_runnable_config(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    seen_thread_ids = []

    async def generate(state: MessageState, config: RunnableConfig):
        seen_thread_ids.append(config["configurable"]["thread_id"])
        return {"messages": [AIMessage(content="ok")]}

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile(checkpointer=sqlite_checkpointer)
    )
    registry = GraphRegistry(
        registry={
            "threaded": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request(
        "threaded",
        metadata={RUN_METADATA_KEY: RUN_ID},
    )

    invocation = await run_langgraph("threaded", request.messages, registry, request)

    assert invocation.output == "ok"
    assert seen_thread_ids == [checkpoint_key("threaded", RUN_ID)]


async def test_interrupt_result_is_returned_before_output_rendering(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    async def output_to_text(output):
        raise AssertionError("interrupt output should not be rendered")

    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
                description="DUMMY",
                output_to_text=output_to_text,
                features={GraphFeature.INTERRUPTS},
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request(
        "interruptible",
        metadata={RUN_METADATA_KEY: RUN_ID},
    )

    invocation = await run_langgraph(
        "interruptible",
        request.messages,
        registry,
        request,
    )

    assert isinstance(invocation.output, LangGraphInterruptBatch)
    assert invocation.output.run_id == RUN_ID
    assert len(invocation.output.interrupts) == 1
    assert invocation.output.interrupts[0].value == DEFAULT_INTERRUPT_PAYLOAD
    assert invocation.custom_events == ()


async def test_interrupt_shape_is_ignored_when_interrupts_disabled(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    class Graph:
        output_channels = ("__interrupt__",)

        async def astream(self, *args, **kwargs):
            yield {
                "type": "values",
                "ns": (),
                "data": {"__interrupt__": ["not-enabled"]},
            }

    async def output_to_text(output):
        return output["__interrupt__"][0]

    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
        description="DUMMY",
        output_to_text=output_to_text,
    )
    run = GraphRun(
        config=graph_config,
        graph=Graph(),
        inputs={},
        context=None,
        runnable_config=None,
        run_id=None,
    )

    invocation = await invoke_run(run)

    assert invocation.output == "not-enabled"
    assert invocation.custom_events == ()


@pytest.mark.parametrize("stream", [False, True])
async def test_parallel_interrupts_are_returned_as_one_durable_batch(
    make_request,
    monkeypatch,
    sqlite_checkpointer: AsyncSqliteSaver,
    stream: bool,
) -> None:
    graph = make_parallel_interrupt_graph(sqlite_checkpointer)
    astream_options = []
    original_astream = graph.astream

    def recording_astream(*args, **kwargs):
        astream_options.append(kwargs)
        return original_astream(*args, **kwargs)

    monkeypatch.setattr(graph, "astream", recording_astream)
    registry = GraphRegistry(
        registry={
            "parallel": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                request_to_input=lambda request, messages: {"answers": []},
                output_to_text=lambda output: str(output["answers"]),
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request(
        "parallel",
        metadata={
            RUN_METADATA_KEY: STREAM_RUN_ID if stream else RUN_ID,
        },
    )

    if stream:
        outputs = [
            event
            async for event in run_langgraph_stream(
                "parallel",
                request.messages,
                registry,
                request,
            )
        ]
        assert len(outputs) == 1
        output = outputs[0]
    else:
        invocation = await run_langgraph(
            "parallel",
            request.messages,
            registry,
            request,
        )
        output = invocation.output

    assert isinstance(output, LangGraphInterruptBatch)
    assert len(output.interrupts) == EXPECTED_PARALLEL_INTERRUPTS
    assert len({item.id for item in output.interrupts}) == EXPECTED_PARALLEL_INTERRUPTS
    assert {item.value["question"] for item in output.interrupts} == {
        "left",
        "right",
    }
    assert len(astream_options) == 1
    assert astream_options[0]["durability"] == "exit"


async def test_interrupt_resumes_after_checkpointer_and_graph_restart(
    make_request,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "interrupt-checkpoints.sqlite"

    def registry(checkpointer: AsyncSqliteSaver) -> GraphRegistry:
        return GraphRegistry(
            registry={
                "interruptible": GraphConfig(
                    graph=make_interrupt_graph(checkpointer=checkpointer),
                    description="DUMMY",
                    features={GraphFeature.INTERRUPTS},
                    run_coordinator=InMemoryRunCoordinator(),
                )
            }
        )

    initial_request = make_request(
        "interruptible",
        metadata={RUN_METADATA_KEY: RUN_ID},
    )
    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as saver:
        paused = await run_langgraph(
            "interruptible",
            initial_request.messages,
            registry(saver),
            initial_request,
        )

    assert isinstance(paused.output, LangGraphInterruptBatch)
    assistant, _finish_reason = response_message(paused.output)
    tool_call = (assistant.tool_calls or [])[0]
    resume_request = make_request(
        "interruptible",
        messages=[
            assistant.model_dump(mode="json", exclude_none=True),
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"resume": "approve"}),
            },
        ],
    )

    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as saver:
        completed = await run_langgraph(
            "interruptible",
            resume_request.messages,
            registry(saver),
            resume_request,
        )

    assert completed.output == "resumed:approve"


async def test_interrupt_enabled_graph_requires_checkpointer(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "broken": GraphConfig(
                graph=make_message_graph("ok"),
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request("broken", metadata={RUN_METADATA_KEY: RUN_ID})

    with pytest.raises(GraphConfigurationError, match="checkpointer"):
        await run_langgraph("broken", request.messages, registry, request)


@pytest.mark.parametrize(
    "checkpointer",
    [
        BaseCheckpointSaver(),
        AsyncReadOnlyCheckpointer(),
        AsyncCheckpointerWithoutPendingWrites(),
        AsyncCheckpointerWithoutDelete(),
    ],
)
async def test_interrupt_checkpointer_must_override_required_async_methods(
    checkpointer: BaseCheckpointSaver,
) -> None:
    config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=checkpointer),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
        run_coordinator=InMemoryRunCoordinator(),
    )

    with pytest.raises(GraphConfigurationError, match="fully asynchronous"):
        await config.resolve_graph()
