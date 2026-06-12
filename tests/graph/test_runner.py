import asyncio
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.constants import TAG_HIDDEN
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest, Role
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class State(TypedDict, total=False):
    question: str
    answer: str
    internal: str


class Input(TypedDict):
    question: str


class Output(TypedDict):
    answer: str


@dataclass
class Context:
    user_id: str


class PydanticState(BaseModel):
    question: str
    answer: str = ""


class PydanticInput(BaseModel):
    question: str


class PydanticOutput(BaseModel):
    answer: str


class RecordingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.starts += 1


def request(model: str, user: str | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[{"role": Role.USER, "content": "question"}],
        user=user,
    )


def registry(name: str, config: GraphConfig) -> GraphRegistry:
    return GraphRegistry(registry={name: config})


async def stream_text(name: str, graph_registry: GraphRegistry) -> str:
    chat_request = request(name)
    chunks = run_langgraph_stream(
        name,
        chat_request.messages,
        graph_registry,
        chat_request,
    )
    return "".join([chunk async for chunk, _ in chunks])


def test_message_defaults_and_callback_list(message_graph) -> None:
    callback = RecordingCallback()
    graph_registry = registry(
        "messages",
        GraphConfig(graph=message_graph, runtime_callbacks=[callback]),
    )
    chat_request = request("messages")

    response, _ = asyncio.run(
        run_langgraph(
            "messages",
            chat_request.messages,
            graph_registry,
            chat_request,
        )
    )

    assert response == "hello"
    assert callback.starts == 1


def test_typed_dict_schemas_and_native_context() -> None:
    model = FakeListChatModel(responses=["answer"])
    output_keys = []

    async def generate(state: State, runtime: Runtime[Context]):
        assert isinstance(runtime.context, Context)
        message = await model.ainvoke([HumanMessage(content=state["question"])])
        return {
            "answer": f"{runtime.context.user_id}:{message.content}",
            "internal": "filtered",
        }

    graph = (
        StateGraph(
            State,
            input_schema=Input,
            output_schema=Output,
            context_schema=Context,
        )
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )

    async def output_to_text(output):
        output_keys.append(set(output))
        return output["answer"]

    graph_registry = registry(
        "typed",
        GraphConfig(
            graph=graph,
            request_to_input=lambda request, messages: {
                "question": messages[-1].content,
                "ignored": True,
            },
            context_factory=lambda request: {"user_id": request.user},
            output_to_text=output_to_text,
        ),
    )
    chat_request = request("typed", user="alice")

    response, _ = asyncio.run(
        run_langgraph(
            "typed",
            chat_request.messages,
            graph_registry,
            chat_request,
        )
    )

    assert response == "alice:answer"
    assert output_keys == [{"answer"}]


def test_pydantic_schemas_and_async_adapters() -> None:
    async def generate(state: PydanticState):
        return {"answer": state.question}

    graph = (
        StateGraph(
            PydanticState,
            input_schema=PydanticInput,
            output_schema=PydanticOutput,
        )
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )

    async def resolve_graph():
        return graph

    async def request_to_input(request, messages):
        return {"question": messages[-1].content, "ignored": True}

    async def context_factory(request):
        return None

    graph_registry = registry(
        "pydantic",
        GraphConfig(
            graph=resolve_graph,
            request_to_input=request_to_input,
            context_factory=context_factory,
            output_to_text=lambda output: output["answer"],
        ),
    )
    chat_request = request("pydantic")

    response, _ = asyncio.run(
        run_langgraph(
            "pydantic",
            chat_request.messages,
            graph_registry,
            chat_request,
        )
    )

    assert response == "question"


def test_nested_subgraph_streaming() -> None:
    model = FakeListChatModel(responses=["nested"])

    async def generate(state: State):
        await model.ainvoke([HumanMessage(content=state["question"])])
        return {"answer": "done"}

    subgraph = (
        StateGraph(State)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    graph = (
        StateGraph(State, input_schema=Input, output_schema=Output)
        .add_node("subgraph", subgraph)
        .set_entry_point("subgraph")
        .set_finish_point("subgraph")
        .compile()
    )
    graph_registry = registry(
        "nested",
        GraphConfig(
            graph=graph,
            request_to_input=lambda request, messages: {
                "question": messages[-1].content
            },
            output_to_text=lambda output: output["answer"],
            streamable_node_names=["generate"],
        ),
    )

    assert asyncio.run(stream_text("nested", graph_registry)) == "nested"


def test_stream_filters_nodes_hidden_tags_and_non_ai_messages() -> None:
    draft_model = FakeListChatModel(responses=["draft"])
    hidden_model = FakeListChatModel(responses=["hidden"]).with_config(
        tags=[TAG_HIDDEN]
    )
    visible_model = FakeListChatModel(responses=["visible"])

    async def draft(state: MessageState):
        return {"messages": [await draft_model.ainvoke(state["messages"])]}

    async def generate(state: MessageState):
        await hidden_model.ainvoke(state["messages"])
        return {"messages": [await visible_model.ainvoke(state["messages"])]}

    builder = StateGraph(MessageState)
    builder.add_node(
        "non_ai",
        lambda state: {"messages": [HumanMessage(content="ignored")]},
    )
    builder.add_node("draft", draft)
    builder.add_node("generate", generate)
    builder.set_entry_point("non_ai")
    builder.add_edge("non_ai", "draft")
    builder.add_edge("draft", "generate")
    builder.set_finish_point("generate")
    graph_registry = registry(
        "filtered",
        GraphConfig(
            graph=builder.compile(),
            streamable_node_names=["non_ai", "generate"],
        ),
    )

    assert asyncio.run(stream_text("filtered", graph_registry)) == "visible"
