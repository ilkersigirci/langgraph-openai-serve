"""Deep-research behavior through the LGOS Responses API with fixture transports."""

import itertools
import json
from base64 import b64encode
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import pytest
from anyio import fail_after
from anyio.lowlevel import checkpoint
from fastapi import FastAPI
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_openai_serve import (
    GraphRegistry,
    InMemoryBackgroundBackend,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from openai import AsyncOpenAI
from openai.types.responses import Response

from lgos_demo_api.graphs.deep_research import (
    ANSWER_NOW,
    START_RESEARCH,
    WEB_SEARCH_NEEDED,
    DeepResearchSettings,
    create_deep_research_graph,
    create_deep_research_graph_config,
)

MODEL = "deep-research"
WEB_SEARCH = [{"type": "web_search"}]
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/interrupts"
# One question means one researcher, so the model replies keep their order.
PLAN = ["How does a LangGraph interrupt pause and resume a run?"]
QUESTION = "How do LangGraph interrupts work?"
# Provider IDs are unique, and add_messages replaces a message with a known ID.
_ids = itertools.count()


def model_response(
    text: str = "",
    *,
    calls: Sequence[dict[str, Any]] = (),
    refusal: str | None = None,
) -> dict[str, Any]:
    """Build a completed Responses result."""
    number = next(_ids)
    content = (
        {"type": "refusal", "refusal": refusal}
        if refusal
        else {"type": "output_text", "text": text, "annotations": []}
    )
    return {
        "id": f"resp_{number}",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-luna",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            *calls,
            {
                "id": f"msg_{number}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [content],
            },
        ],
    }


def triage_response(*, deep: bool, plan: Sequence[str] = PLAN) -> dict[str, Any]:
    return model_response(
        json.dumps({"deep_research": deep, "plan": list(plan) if deep else []})
    )


def search_calls(*queries: str) -> dict[str, Any]:
    """Build a researcher turn that searches every query in parallel."""
    calls = []
    for query in queries:
        call_id = next(_ids)
        calls.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": f"call_{call_id}",
                "name": "web_search",
                "arguments": json.dumps({"query": query}),
                "status": "completed",
            }
        )
    return model_response(calls=calls)


def response_events(response: dict[str, Any]) -> bytes:
    """Encode a Responses result as the SSE events of a streamed call."""
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        }
    ]
    for index, item in enumerate(response["output"]):
        if item["type"] == "function_call":
            added = {**item, "arguments": ""}
            delta = {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "delta": item["arguments"],
            }
        elif (part := item["content"][0])["type"] == "refusal":
            added = {**item, "content": []}
            delta = {
                "type": "response.refusal.done",
                "item_id": item["id"],
                "content_index": 0,
                "refusal": part["refusal"],
            }
        else:
            added = {**item, "content": []}
            delta = {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "content_index": 0,
                "delta": part["text"],
                "logprobs": [],
            }
        events += [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": added,
            },
            {**delta, "output_index": index},
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]
    events.append({"type": "response.completed", "response": response})
    return "".join(
        f"event: {event['type']}\n"
        f"data: {json.dumps({**event, 'sequence_number': number})}\n\n"
        for number, event in enumerate(events)
    ).encode()


class ModelProvider:
    """Serve queued Responses results and record every request payload."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        # Like OpenAI, echo the text format; LangChain parses streamed
        # structured output only when the response declares its JSON schema.
        response = {**self.responses.popleft(), "text": payload.get("text")}
        if not payload.get("stream"):
            return httpx2.Response(200, json=response)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_events(response),
        )


@tool("web_search", response_format="content_and_artifact")
async def fixture_web_search(query: str) -> tuple[str, dict[str, str]]:
    """Search the public web."""
    return f"Results for {query}: {SOURCE_URL}", {SOURCE_URL: "LangGraph interrupts"}


@asynccontextmanager
async def graph_client(
    checkpointer: AsyncSqliteSaver,
    provider: ModelProvider,
    *,
    files_handler: Any = None,
) -> AsyncGenerator[AsyncOpenAI, None]:
    """Serve the graph with an in-process background engine."""

    async def unexpected_file_request(request: httpx2.Request) -> httpx2.Response:
        msg = f"Unexpected Files API request: {request.url}"
        raise AssertionError(msg)

    async with (
        httpx2.AsyncClient(
            transport=httpx2.MockTransport(provider.handle)
        ) as model_http,
        httpx2.AsyncClient(
            transport=httpx2.MockTransport(files_handler or unexpected_file_request)
        ) as files_http,
        AsyncOpenAI(
            api_key="test",
            base_url="http://files.test/v1",
            http_client=files_http,
            max_retries=0,
        ) as files,
    ):
        model = ChatOpenAI(
            model="gpt-6-luna",
            base_url="http://model.test/v1",
            api_key="test",
            use_responses_api=True,
            store=False,
            http_async_client=model_http,
        )
        graph = create_deep_research_graph(
            checkpointer, files, model=model, web_search_tool=fixture_web_search
        )
        registry = GraphRegistry(
            graphs={MODEL: create_deep_research_graph_config(graph)},
            run_coordinator=InMemoryRunCoordinator(),
        )
        background = InMemoryBackgroundBackend(registry)
        app = FastAPI(lifespan=background.lifespan)
        LanggraphOpenaiServe(
            app=app, registry=registry, background=background
        ).bind_openai_api()
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://test"
            ) as http,
            AsyncOpenAI(
                api_key="test",
                base_url="http://test/v1",
                http_client=http,
                max_retries=0,
            ) as client,
        ):
            yield client


def review(paused: Response) -> dict[str, Any]:
    (call,) = (item for item in paused.output if item.type == "function_call")
    assert call.name == "lgos_interrupt"
    return json.loads(call.arguments)


def plan(paused: Response) -> list[str]:
    """Read the plan's bullets from the review question."""
    question = review(paused)["question"]
    lines = question.splitlines()
    return [line.removeprefix("- ") for line in lines if line.startswith("- ")]


def reply_to(paused: Response, reply: str) -> list[dict[str, str]]:
    return [
        {"type": "function_call_output", "call_id": item.call_id, "output": reply}
        for item in paused.output
        if item.type == "function_call"
    ]


async def streamed(client: AsyncOpenAI, **request: Any) -> Response:
    stream = await client.responses.create(**request, stream=True)
    return [event async for event in stream][-1].response


async def finished(client: AsyncOpenAI, **request: Any) -> Response:
    response = await client.responses.create(**request, background=True)
    with fail_after(5):
        while response.status in {"queued", "in_progress"}:
            await checkpoint()
            response = await client.responses.retrieve(response.id)
    return response


def searches(response: Response) -> list[str]:
    return [
        item.action.query for item in response.output if item.type == "web_search_call"
    ]


@pytest.mark.parametrize(
    ("tools", "replies"),
    [
        pytest.param(
            [],
            [triage_response(deep=False), model_response("Hello!")],
            id="web-search-off",
        ),
        pytest.param(
            WEB_SEARCH,
            [triage_response(deep=False), model_response("Hello!")],
            id="simple-request",
        ),
        pytest.param(
            WEB_SEARCH,
            [model_response(refusal="I cannot decide."), model_response("Hello!")],
            id="triage-refusal",
        ),
    ],
)
async def test_requests_without_research_stream_only_a_direct_answer(
    sqlite_checkpointer: AsyncSqliteSaver,
    tools: list[dict[str, str]],
    replies: list[dict[str, Any]],
) -> None:
    provider = ModelProvider(*replies)
    async with graph_client(sqlite_checkpointer, provider) as client:
        final = await streamed(client, model=MODEL, input="Hello", tools=tools)

    assert [(item.type, item.phase) for item in final.output] == [
        ("message", "final_answer")
    ]
    assert final.output_text == "Hello!"
    assert len(provider.requests) == len(replies)


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streamed"])
async def test_research_without_web_search_asks_the_user_to_turn_it_on(
    sqlite_checkpointer: AsyncSqliteSaver,
    stream: bool,
) -> None:
    provider = ModelProvider(triage_response(deep=True))
    request = {"model": MODEL, "input": QUESTION}
    async with graph_client(sqlite_checkpointer, provider) as client:
        response = await (
            streamed(client, **request)
            if stream
            else client.responses.create(**request)
        )

    # Only the client allows web access, so nothing pauses or searches.
    assert [item.type for item in response.output] == ["message"]
    assert response.output_text == WEB_SEARCH_NEEDED
    assert len(provider.requests) == 1


async def test_research_does_not_start_once_web_search_is_turned_off(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    provider = ModelProvider(triage_response(deep=True))
    async with graph_client(sqlite_checkpointer, provider) as client:
        paused = await client.responses.create(
            model=MODEL, input=QUESTION, tools=WEB_SEARCH
        )
        final = await client.responses.create(
            model=MODEL,
            previous_response_id=paused.id,
            input=reply_to(paused, START_RESEARCH),
        )

    assert searches(final) == []
    assert final.output_text == WEB_SEARCH_NEEDED
    assert len(provider.requests) == 1


@pytest.mark.parametrize("background", [False, True], ids=["streamed", "background"])
async def test_approved_research_returns_its_searches_and_a_cited_report(
    sqlite_checkpointer: AsyncSqliteSaver,
    background: bool,
) -> None:
    report = f"Interrupts pause a run; see [LangGraph interrupts]({SOURCE_URL})."
    provider = ModelProvider(
        triage_response(deep=True),
        search_calls("LangGraph interrupt pause", "LangGraph resume"),
        model_response("Both questions have evidence."),
        model_response(report),
    )
    async with graph_client(sqlite_checkpointer, provider) as client:
        paused = await client.responses.create(
            model=MODEL, input=QUESTION, tools=WEB_SEARCH
        )
        resume = {
            "model": MODEL,
            "previous_response_id": paused.id,
            "input": reply_to(paused, START_RESEARCH),
            "tools": WEB_SEARCH,
        }
        final = await (finished if background else streamed)(client, **resume)

    pause = review(paused)
    assert plan(paused) == PLAN
    assert pause["choices"] == [START_RESEARCH, ANSWER_NOW]
    assert pause["allow_other"] is True
    assert final.status == "completed"
    assert searches(final) == ["LangGraph interrupt pause", "LangGraph resume"]
    (answer,) = (
        item
        for item in final.output
        if item.type == "message" and item.phase == "final_answer"
    )
    assert answer.content[0].text == report
    assert answer.content[0].annotations[0].url == SOURCE_URL
    # A model that replays search calls without the tool can write a call into
    # the report as text, so the report request keeps the tool but disables it.
    report_request = provider.requests[-1]
    assert [tool["name"] for tool in report_request["tools"]] == ["web_search"]
    assert report_request["tool_choice"] == "none"
    # Background Responses are polled, so they carry no transient status.
    commentary = [
        item.content[0].text
        for item in final.output
        if item.type == "message" and item.phase == "commentary"
    ]
    assert commentary == (
        []
        if background
        else [
            "Searching the web: LangGraph interrupt pause",
            "Searching the web: LangGraph resume",
            "Writing the report",
        ]
    )


async def test_plan_edits_revise_the_plan_until_the_user_answers_now(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    revised = ["How is a paused LangGraph run resumed?"]
    provider = ModelProvider(
        triage_response(deep=True),
        triage_response(deep=True, plan=revised),
        model_response("Resume it with Command(resume=...)."),
    )
    async with graph_client(sqlite_checkpointer, provider) as client:
        first = await client.responses.create(
            model=MODEL, input=QUESTION, tools=WEB_SEARCH
        )
        second = await client.responses.create(
            model=MODEL,
            previous_response_id=first.id,
            input=reply_to(first, "Only cover resuming."),
            tools=WEB_SEARCH,
        )
        answered = await client.responses.create(
            model=MODEL,
            previous_response_id=second.id,
            input=reply_to(second, ANSWER_NOW),
            tools=WEB_SEARCH,
        )

    assert plan(second) == revised
    revision = json.dumps(provider.requests[1]["input"])
    assert PLAN[0] in revision
    assert "Only cover resuming." in revision
    # The edit only revises the plan; answering now answers the request.
    answer = json.dumps(provider.requests[2]["input"])
    assert QUESTION in answer
    assert "Only cover resuming." not in answer
    assert answered.output_text == "Resume it with Command(resume=...)."
    assert searches(answered) == []


async def test_a_blank_reply_asks_again_with_the_same_plan(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    provider = ModelProvider(triage_response(deep=True))
    async with graph_client(sqlite_checkpointer, provider) as client:
        first = await client.responses.create(
            model=MODEL, input=QUESTION, tools=WEB_SEARCH
        )
        second = await client.responses.create(
            model=MODEL,
            previous_response_id=first.id,
            input=reply_to(first, "  "),
            tools=WEB_SEARCH,
        )

    assert plan(second) == PLAN
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "settings", [{}, {"research_rounds": 1}], ids=["default", "one-round"]
)
async def test_research_ends_after_its_last_round(
    sqlite_checkpointer: AsyncSqliteSaver,
    settings: dict[str, int],
) -> None:
    rounds = DeepResearchSettings.model_validate(settings).research_rounds
    queries = [f"query {number}" for number in range(1, rounds + 1)]
    # The researcher always asks for more; the provider has no reply for a
    # round beyond the limit.
    provider = ModelProvider(
        triage_response(deep=True),
        *(search_calls(query) for query in queries),
        model_response("Report from every round."),
    )
    # Each request carries its own settings, so the resume repeats them.
    metadata = {"lgos_settings": json.dumps(settings)}
    async with graph_client(sqlite_checkpointer, provider) as client:
        paused = await client.responses.create(
            model=MODEL, input=QUESTION, tools=WEB_SEARCH, metadata=metadata
        )
        final = await client.responses.create(
            model=MODEL,
            previous_response_id=paused.id,
            input=reply_to(paused, START_RESEARCH),
            tools=WEB_SEARCH,
            metadata=metadata,
        )

    assert searches(final) == queries
    assert final.output_text == "Report from every round."


async def test_research_starts_without_a_pause_when_plan_review_is_off(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    provider = ModelProvider(
        triage_response(deep=True),
        search_calls("LangGraph interrupts"),
        model_response("The question has evidence."),
        model_response("Report."),
    )
    async with graph_client(sqlite_checkpointer, provider) as client:
        response = await client.responses.create(
            model=MODEL,
            input=QUESTION,
            tools=WEB_SEARCH,
            metadata={"lgos_settings": json.dumps({"review_plan": False})},
        )

    assert not any(item.type == "function_call" for item in response.output)
    assert searches(response) == ["LangGraph interrupts"]
    assert response.output_text == "Report."


async def test_each_plan_question_gets_its_own_parallel_researcher(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    questions = ["How does an interrupt pause a run?", "How is a paused run resumed?"]
    # With one round, each researcher makes one model call, so the two
    # researchers may take these replies in either order.
    provider = ModelProvider(
        triage_response(deep=True, plan=questions),
        search_calls("pause"),
        search_calls("resume"),
        model_response("Report."),
    )
    settings = {"review_plan": False, "research_rounds": 1}
    async with graph_client(sqlite_checkpointer, provider) as client:
        response = await client.responses.create(
            model=MODEL,
            input=QUESTION,
            tools=WEB_SEARCH,
            metadata={"lgos_settings": json.dumps(settings)},
        )

    assert sorted(searches(response)) == ["pause", "resume"]
    assert response.output_text == "Report."
    _, *researchers, report = (json.dumps(request) for request in provider.requests)
    # Each researcher receives its own question and nothing of the other's.
    assert sorted(
        [question for question in questions if question in request]
        for request in researchers
    ) == [[question] for question in sorted(questions)]
    assert "Results for pause" in report
    assert "Results for resume" in report


async def test_required_web_search_researches_even_a_simple_request(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    provider = ModelProvider(triage_response(deep=False))
    async with graph_client(sqlite_checkpointer, provider) as client:
        paused = await client.responses.create(
            model=MODEL, input=QUESTION, tools=WEB_SEARCH, tool_choice="required"
        )

    # Without a plan from triage, the request itself is the research question.
    assert plan(paused) == [QUESTION]


async def test_a_named_client_function_stays_client_owned(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    call = {
        "type": "function_call",
        "id": "fc_order",
        "call_id": "call_order",
        "name": "lookup_order",
        "arguments": '{"order_id":"123"}',
        "status": "completed",
    }
    provider = ModelProvider(model_response(calls=[call]))
    async with graph_client(sqlite_checkpointer, provider) as client:
        response = await client.responses.create(
            model=MODEL,
            input="Look up order 123",
            tools=[
                *WEB_SEARCH,
                {
                    "type": "function",
                    "name": "lookup_order",
                    "description": "Look up an order by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ],
            tool_choice={"type": "function", "name": "lookup_order"},
            parallel_tool_calls=False,
        )

    (request,) = provider.requests
    assert [tool["name"] for tool in request["tools"]] == ["lookup_order"]
    assert request["tool_choice"] == {"type": "function", "name": "lookup_order"}
    assert request["parallel_tool_calls"] is False
    (output_call,) = (item for item in response.output if item.type == "function_call")
    assert (output_call.call_id, output_call.name, output_call.arguments) == (
        "call_order",
        "lookup_order",
        '{"order_id":"123"}',
    )


async def test_attachments_reach_triage_and_report_but_never_research(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    content = b"The project codename is BLUE-HERON."

    async def handle_files(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/content"):
            return httpx2.Response(
                200, content=content, headers={"content-type": "text/plain"}
            )
        return httpx2.Response(
            200,
            json={
                "id": "file_brief",
                "object": "file",
                "bytes": len(content),
                "created_at": 1,
                "filename": "brief.txt",
                "purpose": "user_data",
                "status": "processed",
            },
        )

    provider = ModelProvider(
        triage_response(deep=True),
        search_calls("LangGraph interrupts"),
        model_response("The question has evidence."),
        model_response("Report."),
    )
    async with graph_client(
        sqlite_checkpointer, provider, files_handler=handle_files
    ) as client:
        paused = await client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": QUESTION},
                        {"type": "input_file", "file_id": "file_brief"},
                    ],
                }
            ],
            tools=WEB_SEARCH,
        )
        await client.responses.create(
            model=MODEL,
            previous_response_id=paused.id,
            input=reply_to(paused, START_RESEARCH),
            tools=WEB_SEARCH,
        )

    file_data = "data:text/plain;base64," + b64encode(content).decode()
    triage, *research, report = (json.dumps(request) for request in provider.requests)
    assert file_data in triage
    assert file_data in report
    assert len(research) == 2
    assert all(file_data not in request for request in research)
    assert all(QUESTION not in request for request in research)
