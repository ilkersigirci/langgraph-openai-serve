import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from anyio import (
    Event,
    create_task_group,
    fail_after,
    get_cancelled_exc_class,
    sleep_forever,
)
from anyio.lowlevel import checkpoint
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.types import (
    GraphOutput,
    Interrupt,
    StateSnapshot,
    ValuesStreamPart,
    interrupt,
)
from pydantic import ValidationError

from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.interrupt import (
    InMemoryRunCoordinator,
    InterruptResume,
    LangGraphInterruptBatch,
    RunLease,
)
from langgraph_openai_serve.graph.interrupt.state import checkpoint_key
from langgraph_openai_serve.graph.runner import (
    invoke_run,
    run_langgraph,
    run_langgraph_stream,
    stream_run,
)
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    prepare_run,
)
from langgraph_openai_serve.protocol import RUN_METADATA_KEY
from tests.graph.support.interrupt import (
    DEFAULT_INTERRUPT_PAYLOAD,
    make_interrupt_graph,
    make_parallel_interrupt_graph,
    make_parallel_nested_interrupt_graph,
    make_sequential_nested_interrupt_graph,
)
from tests.graph.support.message import make_message_graph
from tests.graph.support.schemas import MessageState

EXPECTED_PARALLEL_INTERRUPTS = 2
SHA256_HEX_LENGTH = 64
RUN_ID = "11111111-1111-4111-8111-111111111111"


class AsyncReadOnlyCheckpointer(BaseCheckpointSaver):
    async def aget_tuple(self, config: RunnableConfig):
        return None

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # ruff: ignore[builtin-argument-shadowing]
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items: tuple[CheckpointTuple, ...] = ()
        for item in items:
            yield item


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


class AsyncCheckpointerWithoutList(AsyncCheckpointerWithoutPendingWrites):
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return None

    alist = BaseCheckpointSaver.alist


class AsyncCheckpointerWithoutDelete(AsyncCheckpointerWithoutList):
    alist = AsyncReadOnlyCheckpointer.alist
    adelete_thread = BaseCheckpointSaver.adelete_thread


async def test_cancelled_preparation_finishes_lease_release(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    state_read_started = Event()
    release_started = Event()
    released = Event()
    cancellation_propagated = Event()

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
                request,
                [HumanMessage(content="question")],
                registry,
            )
        except get_cancelled_exc_class():
            cancellation_propagated.set()

    with fail_after(1):
        async with create_task_group() as task_group:
            task_group.start_soon(run_preparation)
            await state_read_started.wait()
            task_group.cancel_scope.cancel()

    assert release_started.is_set()
    assert released.is_set()
    assert cancellation_propagated.is_set()


async def test_lost_execution_lease_preserves_checkpoint_thread(make_request) -> None:
    class RecordingSaver(InMemorySaver):
        def __init__(self) -> None:
            super().__init__()
            self.deleted_threads: list[str] = []

        async def adelete_thread(self, thread_id: str) -> None:
            self.deleted_threads.append(thread_id)
            await super().adelete_thread(thread_id)

    saver = RecordingSaver()
    lease = RunLease()
    owner: asyncio.Task[object] | None = None

    @asynccontextmanager
    async def coordinator(_key: str):
        nonlocal owner
        owner = asyncio.current_task()
        yield lease

    async def lose_lease(_state: MessageState):
        lease.lost = True
        assert owner is not None
        owner.cancel()
        await sleep_forever()

    graph = (
        StateGraph(MessageState)
        .add_node("lose_lease", lose_lease)
        .set_entry_point("lose_lease")
        .set_finish_point("lose_lease")
        .compile(checkpointer=saver)
    )
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

    with pytest.raises(asyncio.CancelledError):
        await run_langgraph(
            make_request(
                "interruptible",
                metadata={RUN_METADATA_KEY: RUN_ID},
            ),
            [HumanMessage(content="question")],
            registry,
        )

    assert saver.deleted_threads == []


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

    message = await run_langgraph(request, [HumanMessage(content="question")], registry)

    assert isinstance(message, AIMessage)
    assert message.text == "ok"
    assert seen_thread_ids == [checkpoint_key("threaded", RUN_ID)]


async def test_interrupt_result_is_returned_before_output_rendering(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    async def output_to_message(output):
        msg = "interrupt output should not be rendered"
        raise AssertionError(msg)

    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
                description="DUMMY",
                output_to_message=output_to_message,
                features={GraphFeature.INTERRUPTS},
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request(
        "interruptible",
        metadata={RUN_METADATA_KEY: RUN_ID},
    )

    batch = await run_langgraph(request, [HumanMessage(content="question")], registry)

    assert isinstance(batch, LangGraphInterruptBatch)
    assert batch.run_id == RUN_ID
    assert len(batch.interrupts) == 1
    assert batch.interrupts[0].value == DEFAULT_INTERRUPT_PAYLOAD


@pytest.mark.parametrize("stream", [False, True], ids=["invoke", "stream"])
async def test_undeclared_interrupt_cannot_be_rendered_as_success(
    make_request,
    stream: bool,
) -> None:
    def ask(_state: MessageState):
        interrupt({"question": "Approve?"})

    graph = (
        StateGraph(MessageState)
        .add_node("ask", ask)
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile()
    )
    registry = GraphRegistry(
        registry={
            "undeclared-interrupt": GraphConfig(
                graph=graph,
                description="An interrupt without the required feature declaration.",
                output_to_message=lambda _output: AIMessage(content="success"),
            )
        }
    )
    request = make_request("undeclared-interrupt")
    messages = [HumanMessage(content="question")]

    async def execute() -> None:
        if stream:
            _ = [
                event
                async for event in run_langgraph_stream(request, messages, registry)
            ]
        else:
            await run_langgraph(request, messages, registry)

    with pytest.raises(GraphConfigurationError, match=r"GraphFeature\.INTERRUPTS"):
        await execute()


async def test_interrupt_shape_is_ignored_when_interrupts_disabled(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    class Graph:
        output_channels = ("__interrupt__",)

        async def ainvoke(self, *args, **kwargs):
            return GraphOutput(value={"__interrupt__": ["not-enabled"]})

    async def output_to_message(output):
        return AIMessage(content=output["__interrupt__"][0])

    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
        description="DUMMY",
        output_to_message=output_to_message,
    )
    run = GraphRun(
        config=graph_config,
        graph=Graph(),
        inputs={},
        context=None,
        runnable_config=None,
        run_id=None,
    )

    async with run:
        message = await invoke_run(run)

    assert isinstance(message, AIMessage)
    assert message.text == "not-enabled"


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(False, id="non-streaming"),
        pytest.param(True, id="streaming"),
    ],
)
async def test_parallel_interrupts_are_returned_as_one_durable_batch(
    make_request,
    monkeypatch,
    sqlite_checkpointer: AsyncSqliteSaver,
    stream: bool,
) -> None:
    graph = make_parallel_interrupt_graph(sqlite_checkpointer)
    astream_options = []
    ainvoke_options = []
    original_astream = graph.astream
    original_ainvoke = graph.ainvoke

    def recording_astream(*args, **kwargs):
        astream_options.append(kwargs)
        return original_astream(*args, **kwargs)

    async def recording_ainvoke(*args, **kwargs):
        ainvoke_options.append(kwargs)
        return await original_ainvoke(*args, **kwargs)

    monkeypatch.setattr(graph, "astream", recording_astream)
    monkeypatch.setattr(graph, "ainvoke", recording_ainvoke)
    registry = GraphRegistry(
        registry={
            "parallel": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_message=lambda output: AIMessage(
                    content=str(output["answers"])
                ),
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request(
        "parallel",
        metadata={RUN_METADATA_KEY: RUN_ID},
    )

    if stream:
        outputs = [
            event
            async for event in run_langgraph_stream(
                request, [HumanMessage(content="question")], registry
            )
        ]
        assert len(outputs) == 1
        output = outputs[0]
    else:
        output = await run_langgraph(
            request, [HumanMessage(content="question")], registry
        )

    assert isinstance(output, LangGraphInterruptBatch)
    assert len(output.interrupts) == EXPECTED_PARALLEL_INTERRUPTS
    assert len({item.id for item in output.interrupts}) == EXPECTED_PARALLEL_INTERRUPTS
    assert {item.value["question"] for item in output.interrupts} == {
        "left",
        "right",
    }
    options = astream_options[0] if stream else ainvoke_options[0]
    assert options["durability"] == "exit"


@pytest.mark.parametrize(
    ("graph_factory", "expected_questions"),
    [
        pytest.param(
            make_parallel_nested_interrupt_graph,
            {"nested-a", "nested-b"},
            id="nested-parallel",
        ),
        pytest.param(
            make_sequential_nested_interrupt_graph,
            {"first"},
            id="indirectly-nested",
        ),
    ],
)
async def test_stream_returns_nested_interrupts_from_root_values(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
    graph_factory,
    expected_questions: set[str],
) -> None:
    graph = graph_factory(sqlite_checkpointer)
    registry = GraphRegistry(
        registry={
            "nested": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_message=lambda output: AIMessage(
                    content=str(output["answers"])
                ),
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    request = make_request(
        "nested",
        metadata={RUN_METADATA_KEY: RUN_ID},
    )

    outputs = [
        event
        async for event in run_langgraph_stream(
            request,
            [HumanMessage(content="question")],
            registry,
        )
    ]

    assert len(outputs) == 1
    batch = outputs[0]
    assert isinstance(batch, LangGraphInterruptBatch)
    assert {interrupt.value["question"] for interrupt in batch.interrupts} == (
        expected_questions
    )


async def test_stream_rejects_conflicting_duplicate_interrupt_id(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    graph = make_interrupt_graph(checkpointer=sqlite_checkpointer)
    first = Interrupt(value={"question": "first"}, id="duplicate")
    conflicting = Interrupt(value={"question": "second"}, id="duplicate")

    async def graph_events(*_args, **_kwargs):
        yield ValuesStreamPart(
            type="values",
            ns=(),
            data={"messages": []},
            interrupts=(first,),
        )
        yield ValuesStreamPart(
            type="values",
            ns=(),
            data={"messages": []},
            interrupts=(first,),
        )
        yield ValuesStreamPart(
            type="values",
            ns=(),
            data={"messages": []},
            interrupts=(conflicting,),
        )

    monkeypatch.setattr(graph, "astream", graph_events)
    run = GraphRun(
        config=GraphConfig(
            graph=graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            run_coordinator=InMemoryRunCoordinator(),
        ),
        graph=graph,
        inputs={},
        context=None,
        runnable_config={"configurable": {"thread_id": "conflicting-interrupts"}},
        run_id=RUN_ID,
        checkpoint_thread_id="conflicting-interrupts",
    )

    with pytest.raises(RuntimeError, match="conflicting data"):
        async with run:
            _ = [event async for event in stream_run(run)]


async def test_durable_state_rejects_duplicate_interrupt_id(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    graph = make_interrupt_graph(checkpointer=sqlite_checkpointer)
    duplicate = Interrupt(value={"question": "Approve?"}, id="duplicate")

    async def duplicate_state(*_args, **_kwargs):
        return StateSnapshot(
            values={},
            next=("ask",),
            config={
                "configurable": {
                    "thread_id": "duplicate-interrupts",
                    "checkpoint_id": "checkpoint",
                }
            },
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(duplicate, duplicate),
        )

    monkeypatch.setattr(graph, "aget_state", duplicate_state)
    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )

    with pytest.raises(RuntimeError, match="duplicate interrupt ids"):
        await prepare_run(
            make_request(
                "interruptible",
                metadata={RUN_METADATA_KEY: RUN_ID},
            ),
            [HumanMessage(content="question")],
            registry,
        )


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
            initial_request, [HumanMessage(content="question")], registry(saver)
        )

    assert isinstance(paused, LangGraphInterruptBatch)
    resume = InterruptResume(
        run_id=paused.run_id,
        generation_token=paused.generation_token,
        values={paused.interrupts[0].id: "approve"},
    )

    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as saver:
        completed = await run_langgraph(
            make_request("interruptible"),
            [],
            registry(saver),
            resume=resume,
        )

    assert isinstance(completed, AIMessage)
    assert completed.text == "resumed:approve"


async def test_interrupt_enabled_graph_requires_checkpointer() -> None:
    config = GraphConfig(
        graph=make_message_graph("ok"),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
        run_coordinator=InMemoryRunCoordinator(),
    )

    with pytest.raises(GraphConfigurationError, match="checkpointer"):
        await config.resolve_graph()


def test_interrupt_enabled_graph_requires_run_coordinator() -> None:
    with pytest.raises(ValidationError, match="run_coordinator"):
        GraphConfig(
            graph=make_message_graph("ok"),
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
        )


@pytest.mark.parametrize(
    "checkpointer_type",
    [
        pytest.param(BaseCheckpointSaver, id="base"),
        pytest.param(AsyncReadOnlyCheckpointer, id="read-only"),
        pytest.param(
            AsyncCheckpointerWithoutPendingWrites,
            id="missing-pending-writes",
        ),
        pytest.param(AsyncCheckpointerWithoutList, id="missing-list"),
        pytest.param(AsyncCheckpointerWithoutDelete, id="missing-thread-deletion"),
    ],
)
async def test_interrupt_checkpointer_must_override_required_async_methods(
    checkpointer_type: type[BaseCheckpointSaver],
) -> None:
    checkpointer = checkpointer_type()
    config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=checkpointer),
        description="DUMMY",
        features={GraphFeature.INTERRUPTS},
        run_coordinator=InMemoryRunCoordinator(),
    )

    with pytest.raises(GraphConfigurationError, match="fully asynchronous"):
        await config.resolve_graph()
