"""Demo graph for custom input, output, and runtime context adapters."""

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph_openai_serve import ClientSettings, GraphConfig, GraphRequest


@dataclass(frozen=True)
class AppContext:
    user_id: str


class Input(TypedDict):
    question: str


class Output(TypedDict):
    answer: str


class State(TypedDict, total=False):
    question: str
    answer: str


def generate(state: State, runtime: Runtime[AppContext]) -> Output:
    user_id = runtime.context.user_id
    question = state["question"]
    return {"answer": f"{user_id} asked: {question}"}


custom_io_graph = (
    StateGraph(
        State,  # ty: ignore[invalid-argument-type] - ty does not match TypedDict classes to LangGraph's StateLike bound.
        input_schema=Input,  # ty: ignore[invalid-argument-type] - ty does not match TypedDict classes to LangGraph's StateLike bound.
        output_schema=Output,  # ty: ignore[invalid-argument-type] - ty does not match TypedDict classes to LangGraph's StateLike bound.
        context_schema=AppContext,
    )
    .add_node("generate", generate)
    .set_entry_point("generate")
    .set_finish_point("generate")
    .compile()
)


def request_to_input(
    _request: GraphRequest,
    messages: list[BaseMessage],
) -> Input:
    return {"question": messages[-1].text}


def context_factory(
    request: GraphRequest,
    _client_settings: ClientSettings | None,
) -> AppContext:
    return AppContext(user_id=request.user or "anonymous")


def output_to_message(output: Output) -> AIMessage:
    return AIMessage(content=output["answer"])


custom_io_graph_config = GraphConfig(
    graph=custom_io_graph,
    description=(
        "Demonstrates custom input, output, and typed runtime-context adapters."
    ),
    request_to_input=request_to_input,
    context_factory=context_factory,
    output_to_message=output_to_message,
)
