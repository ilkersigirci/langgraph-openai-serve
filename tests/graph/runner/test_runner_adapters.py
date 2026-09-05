from dataclasses import dataclass

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langgraph_openai_serve import GraphRequest
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream
from tests.graph.support.schemas import (
    AnswerOutput,
    PydanticAnswerOutput,
    PydanticQuestionInput,
    PydanticQuestionState,
    QuestionInput,
    QuestionState,
)


@dataclass
class UserContext:
    user_id: str


async def test_typed_dict_schemas_and_native_context(
    make_request,
) -> None:
    model = FakeListChatModel(responses=["answer"])
    output_keys = []
    adapter_requests = []

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

    async def output_to_message(output):
        output_keys.append(set(output))
        return AIMessage(content=output["answer"])

    def request_to_input(request, messages):
        adapter_requests.append(request)
        return {
            "question": messages[-1].content,
            "ignored": True,
        }

    graph_registry = GraphRegistry(
        registry={
            "typed": GraphConfig(
                graph=graph,
                description="DUMMY",
                request_to_input=request_to_input,
                context_factory=lambda request, _settings: {"user_id": request.user},
                output_to_message=output_to_message,
            )
        },
    )
    request = make_request("typed", user="alice")

    invocation = await run_langgraph(
        request, [HumanMessage(content="question")], graph_registry
    )

    assert invocation.output.text == "alice:answer"
    assert output_keys == [{"answer"}]
    assert len(adapter_requests) == 1
    assert isinstance(adapter_requests[0], GraphRequest)
    assert adapter_requests[0].user == "alice"


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
                description="DUMMY",
                request_to_input=request_to_input,
                context_factory=context_factory,
                output_to_message=lambda output: AIMessage(content=output.answer),
            )
        },
    )
    request = make_request("pydantic")

    invocation = await run_langgraph(
        request, [HumanMessage(content="question")], graph_registry
    )

    assert invocation.output.text == "question"


async def test_stream_and_invoke_render_the_same_output_shape(make_request) -> None:
    output_keys = []

    async def generate(state: QuestionState):
        return {"answer": state["question"], "internal": "filtered"}

    graph = (
        StateGraph(
            QuestionState,
            input_schema=QuestionInput,
            output_schema=AnswerOutput,
        )
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )

    async def output_to_message(output):
        output_keys.append(set(output))
        return AIMessage(content=output["answer"])

    graph_registry = GraphRegistry(
        registry={
            "typed": GraphConfig(
                graph=graph,
                description="DUMMY",
                request_to_input=lambda _request, messages: {
                    "question": messages[-1].content
                },
                output_to_message=output_to_message,
            )
        },
    )
    request = make_request("typed")

    invocation = await run_langgraph(
        request, [HumanMessage(content="question")], graph_registry
    )
    events = [
        event
        async for event in run_langgraph_stream(
            request, [HumanMessage(content="question")], graph_registry
        )
    ]

    assert invocation.output.text == "question"
    assert isinstance(events[-1], AIMessage)
    assert events[-1].text == "question"
    assert output_keys == [{"answer"}, {"answer"}]
