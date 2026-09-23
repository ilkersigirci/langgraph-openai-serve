from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from langgraph_openai_serve.graph.runner import (
    BackgroundCheckpointIncompleteError,
    run_background_graph,
)


class BackgroundState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    private: str


class AnswerOutput(TypedDict):
    answer: str


class PydanticBackgroundState(BaseModel):
    answer: str = ""
    private: str = ""


class PydanticAnswerOutput(BaseModel):
    answer: str


class _InterruptedCheckpointSaver(AsyncSqliteSaver):
    _interrupted = False

    async def aput(self, config, checkpoint, metadata, new_versions):
        if metadata["step"] == 1 and not self._interrupted:
            self._interrupted = True
            message = "checkpoint write interrupted"
            raise OSError(message)
        return await super().aput(config, checkpoint, metadata, new_versions)


async def test_pending_writes_are_resumed_before_background_completion(make_request):
    calls = []

    async def first(_state: BackgroundState):
        calls.append("first")
        return {"answer": "intermediate"}

    async def second(_state: BackgroundState):
        calls.append("second")
        return {"answer": "final"}

    async with _InterruptedCheckpointSaver.from_conn_string(":memory:") as saver:
        graph = (
            StateGraph(BackgroundState, output_schema=AnswerOutput)
            .add_node("first", first)
            .add_node("second", second)
            .set_entry_point("first")
            .add_edge("first", "second")
            .set_finish_point("second")
            .compile(checkpointer=saver)
        )
        config = GraphConfig(
            graph=graph,
            description="Checkpoint recovery",
            background_version="v1",
            output_to_message=lambda output: AIMessage(content=output["answer"]),
            run_coordinator=InMemoryRunCoordinator(),
        )
        request = make_request("background")
        arguments = {
            "checkpoint_thread_id": "pending-writes",
        }
        with pytest.raises(OSError, match="checkpoint write interrupted"):
            await run_background_graph(
                request, [], config, finalize_only=False, **arguments
            )
        with pytest.raises(BackgroundCheckpointIncompleteError):
            await run_background_graph(
                request, [], config, finalize_only=True, **arguments
            )

        recovered = await run_background_graph(
            request, [], config, finalize_only=False, **arguments
        )

    assert recovered.message.text == "final"
    assert calls == ["first", "second"]


async def test_background_recovery_renders_only_declared_output_channels(
    make_request,
    sqlite_checkpointer,
) -> None:
    async def generate(_state: BackgroundState):
        return {
            "messages": [AIMessage(content="durable answer")],
            "answer": "public answer",
            "private": "checkpoint only",
        }

    graph = (
        StateGraph(BackgroundState, output_schema=AnswerOutput)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile(checkpointer=sqlite_checkpointer)
    )
    rendered_outputs: list[object] = []

    def render(output: object) -> AIMessage:
        rendered_outputs.append(output)
        assert output == {"answer": "public answer"}
        return AIMessage(content="rendered answer")

    config = GraphConfig(
        graph=graph,
        description="DUMMY",
        background_version="v1",
        request_to_input=lambda _request, messages: {"messages": messages},
        output_to_message=render,
        run_coordinator=InMemoryRunCoordinator(),
    )
    request = make_request("background")
    messages = [HumanMessage(content="question")]

    executed = await run_background_graph(
        request,
        messages,
        config,
        checkpoint_thread_id="background-output-filter",
        finalize_only=False,
    )
    recovered = await run_background_graph(
        request,
        messages,
        config,
        checkpoint_thread_id="background-output-filter",
        finalize_only=True,
    )

    assert executed.message.text == recovered.message.text == "rendered answer"
    assert rendered_outputs == [
        {"answer": "public answer"},
        {"answer": "public answer"},
    ]
    assert [message.text for message in executed.root_messages] == [
        "question",
        "durable answer",
    ]
    assert recovered.root_messages == executed.root_messages


async def test_background_recovery_restores_pydantic_output_schema(
    make_request,
    sqlite_checkpointer,
) -> None:
    graph = (
        StateGraph(
            PydanticBackgroundState,
            output_schema=PydanticAnswerOutput,
        )
        .add_node(
            "generate",
            lambda _state: {"answer": "public answer", "private": "checkpoint only"},
        )
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile(checkpointer=sqlite_checkpointer)
    )
    rendered_outputs: list[PydanticAnswerOutput] = []

    def render(output: PydanticAnswerOutput) -> AIMessage:
        assert isinstance(output, PydanticAnswerOutput)
        rendered_outputs.append(output)
        return AIMessage(content=output.answer)

    config = GraphConfig(
        graph=graph,
        description="DUMMY",
        background_version="v1",
        request_to_input=lambda _request, _messages: {},
        output_to_message=render,
        run_coordinator=InMemoryRunCoordinator(),
    )

    result = await run_background_graph(
        make_request("background"),
        [],
        config,
        checkpoint_thread_id="background-pydantic-output",
        finalize_only=False,
    )

    assert result.message.text == "public answer"
    assert rendered_outputs == [PydanticAnswerOutput(answer="public answer")]
