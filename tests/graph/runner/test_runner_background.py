from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph_openai_serve import BackgroundPolicy, GraphConfig
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.runner import run_background_graph


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
        background=BackgroundPolicy(version="v1"),
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
        initial_message_count=len(messages),
    )
    recovered = await run_background_graph(
        request,
        messages,
        config,
        checkpoint_thread_id="background-output-filter",
        finalize_only=True,
        initial_message_count=len(messages),
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
        background=BackgroundPolicy(version="v1"),
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
        initial_message_count=0,
    )

    assert result.message.text == "public answer"
    assert rendered_outputs == [PydanticAnswerOutput(answer="public answer")]
