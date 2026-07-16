from dataclasses import dataclass

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph
from tests.graph.support.schemas import (
    AnswerOutput,
    PydanticAnswerOutput,
    PydanticQuestionInput,
    PydanticQuestionState,
    QuestionInput,
    QuestionState,
)

pytestmark = pytest.mark.anyio


@dataclass
class UserContext:
    user_id: str


async def test_typed_dict_schemas_and_native_context(
    make_request,
) -> None:
    model = FakeListChatModel(responses=["answer"])
    output_keys = []

    async def generate(state: QuestionState, runtime: Runtime[UserContext]):
        assert isinstance(runtime.context, UserContext)
        message = await model.ainvoke([HumanMessage(content=state["question"])])
        return {
            "answer": f"{runtime.context.user_id}:{message.content}",
            "internal": "filtered",
        }

    graph = (
        StateGraph(
            QuestionState,
            input_schema=QuestionInput,
            output_schema=AnswerOutput,
            context_schema=UserContext,
        )
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )

    async def output_to_text(output):
        output_keys.append(set(output))
        return output["answer"]

    graph_registry = GraphRegistry(
        registry={
            "typed": GraphConfig(
                graph=graph,
                request_to_input=lambda request, messages: {
                    "question": messages[-1].content,
                    "ignored": True,
                },
                context_factory=lambda request, _settings: {"user_id": request.user},
                output_to_text=output_to_text,
            )
        },
    )
    chat_request = make_request("typed", user="alice")

    invocation = await run_langgraph(
        "typed",
        chat_request.messages,
        graph_registry,
        chat_request,
    )

    assert invocation.output == "alice:answer"
    assert output_keys == [{"answer"}]


async def test_async_graph_factory_and_async_adapters(
    make_request,
) -> None:
    async def generate(state: PydanticQuestionState):
        return {"answer": state.question}

    graph = (
        StateGraph(
            PydanticQuestionState,
            input_schema=PydanticQuestionInput,
            output_schema=PydanticAnswerOutput,
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

    async def context_factory(request, _settings):
        return None

    graph_registry = GraphRegistry(
        registry={
            "pydantic": GraphConfig(
                graph=resolve_graph,
                request_to_input=request_to_input,
                context_factory=context_factory,
                output_to_text=lambda output: output["answer"],
            )
        },
    )
    chat_request = make_request("pydantic")

    invocation = await run_langgraph(
        "pydantic",
        chat_request.messages,
        graph_registry,
        chat_request,
    )

    assert invocation.output == "question"
