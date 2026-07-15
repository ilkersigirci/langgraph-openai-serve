"""Demo graph for custom input, output, and runtime context adapters."""

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest


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


async def generate(state: State, runtime: Runtime[AppContext]) -> Output:
    user_id = runtime.context.user_id
    question = state["question"]
    return {"answer": f"{user_id} asked: {question}"}


custom_io_graph = (
    StateGraph(
        State,
        input_schema=Input,
        output_schema=Output,
        context_schema=AppContext,
    )
    .add_node("generate", generate)
    .set_entry_point("generate")
    .set_finish_point("generate")
    .compile()
)


def request_to_input(
    request: ChatCompletionRequest,
    messages: list[BaseMessage],
) -> Input:
    last_message = messages[-1]
    return {"question": str(last_message.content or "")}


def context_factory(request: ChatCompletionRequest) -> AppContext:
    return AppContext(user_id=request.user or "anonymous")


def output_to_text(output: Output) -> str:
    return output["answer"]


custom_io_graph_config = GraphConfig(
    graph=custom_io_graph,
    request_to_input=request_to_input,
    context_factory=context_factory,
    output_to_text=output_to_text,
)
