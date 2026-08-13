import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from tests.graph.support.schemas import MessageState

DEFAULT_INTERRUPT_PAYLOAD = {"question": "Approve?"}


class InterruptAnswerState(TypedDict, total=False):
    answers: Annotated[list[str], operator.add]


def make_interrupt_graph(
    payload: dict[str, Any] | None = None,
    *,
    checkpointer: BaseCheckpointSaver,
    response_prefix: str = "resumed",
) -> Any:
    interrupt_payload = DEFAULT_INTERRUPT_PAYLOAD if payload is None else payload

    def ask(state: MessageState):
        answer = interrupt(interrupt_payload)
        return {"messages": [AIMessage(content=f"{response_prefix}:{answer}")]}

    return (
        StateGraph(MessageState)
        .add_node("ask", ask)
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile(checkpointer=checkpointer)
    )


def make_parallel_interrupt_graph(checkpointer: BaseCheckpointSaver) -> Any:
    def ask(question: str):
        def node(_state: InterruptAnswerState) -> dict[str, list[str]]:
            return {"answers": [str(interrupt({"question": question}))]}

        return node

    return (
        StateGraph(InterruptAnswerState)
        .add_node("left", ask("left"))
        .add_node("right", ask("right"))
        .add_edge(START, "left")
        .add_edge(START, "right")
        .add_edge("left", END)
        .add_edge("right", END)
        .compile(checkpointer=checkpointer)
    )


def _sequential_interrupt_graph() -> StateGraph:
    def ask_twice(_state: InterruptAnswerState) -> dict[str, list[str]]:
        first = interrupt({"question": "first"})
        second = interrupt({"question": "second"})
        return {"answers": [str(first), str(second)]}

    return (
        StateGraph(InterruptAnswerState)
        .add_node("ask_twice", ask_twice)
        .add_edge(START, "ask_twice")
        .add_edge("ask_twice", END)
    )


def make_sequential_interrupt_graph(checkpointer: BaseCheckpointSaver) -> Any:
    return _sequential_interrupt_graph().compile(checkpointer=checkpointer)


def make_sequential_nested_interrupt_graph(
    checkpointer: BaseCheckpointSaver,
) -> Any:
    nested = _sequential_interrupt_graph().compile()

    async def invoke_nested(state: InterruptAnswerState) -> Any:
        return await nested.ainvoke(state)

    return (
        StateGraph(InterruptAnswerState)
        .add_node("nested", invoke_nested)
        .add_edge(START, "nested")
        .add_edge("nested", END)
        .compile(checkpointer=checkpointer)
    )


def make_parallel_nested_interrupt_graph(
    checkpointer: BaseCheckpointSaver,
) -> Any:
    def subgraph(question: str):
        def ask(_state: InterruptAnswerState) -> dict[str, list[str]]:
            answer = interrupt({"question": question})
            return {"answers": [f"{question}:{answer}"]}

        return (
            StateGraph(InterruptAnswerState)
            .add_node("ask", ask)
            .add_edge(START, "ask")
            .add_edge("ask", END)
            .compile()
        )

    return (
        StateGraph(InterruptAnswerState)
        .add_node("nested_a", subgraph("nested-a"))
        .add_node("nested_b", subgraph("nested-b"))
        .add_edge(START, "nested_a")
        .add_edge(START, "nested_b")
        .add_edge("nested_a", END)
        .add_edge("nested_b", END)
        .compile(checkpointer=checkpointer)
    )
