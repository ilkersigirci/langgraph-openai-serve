"""Advanced graph integration tests with real adapters and fixture transports."""

import json
from base64 import b64encode
from collections import deque
from contextlib import aclosing, asynccontextmanager

import httpx2
import pytest
from anyio import Event, fail_after
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import stream_response
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from langgraph_openai_serve.graph.utils import prepare_run
from openai import AsyncOpenAI, BadRequestError, ConflictError, InternalServerError

from lgos_demo_api.graphs.advanced_graph import (
    OpenAICompatibleKnowledgeBase,
    create_advanced_graph,
    create_advanced_graph_config,
    create_model,
)
from lgos_demo_api.graphs.advanced_graph.knowledge import KnowledgeResult

WEB_TOOL = [{"type": "web_search"}]
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/interrupts"


def model_response(text="", *, calls=(), refusal=None, incomplete=False):
    content = (
        [{"type": "refusal", "refusal": refusal}]
        if refusal
        else [{"type": "output_text", "text": text, "annotations": []}]
    )
    return {
        "id": "resp_provider",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.4-mini",
        "status": "incomplete" if incomplete else "completed",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "output": [
            *calls,
            {
                "id": "msg_provider",
                "type": "message",
                "role": "assistant",
                "status": "incomplete" if incomplete else "completed",
                "content": content,
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def intent_response(intent):
    return model_response(
        calls=[
            {
                "type": "function_call",
                "id": "fc_intent",
                "call_id": "call_intent",
                "name": "IntentDecision",
                "arguments": json.dumps({"intent": intent}),
                "status": "completed",
            }
        ]
    )


def response_events(response):
    events = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        }
    ]
    for index, item in enumerate(response["output"]):
        if item["type"] == "function_call":
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": {**item, "arguments": ""},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": index,
                        "item_id": item["id"],
                        "delta": item["arguments"],
                    },
                ]
            )
        elif item["type"] == "message":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "content": []},
                }
            )
            part = item["content"][0]
            event_type = (
                "response.output_text.delta"
                if part["type"] == "output_text"
                else "response.refusal.done"
            )
            value = part.get("text", part.get("refusal", ""))
            events.append(
                {
                    "type": event_type,
                    "output_index": index,
                    "item_id": item["id"],
                    "content_index": 0,
                    "delta" if part["type"] == "output_text" else "refusal": value,
                    **({"logprobs": []} if part["type"] == "output_text" else {}),
                }
            )
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    events.append({"type": f"response.{response['status']}", "response": response})
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
        for index, event in enumerate(events)
    ).encode()


class ModelProvider:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []

    async def handle(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        assert request.url.path.endswith("/responses")
        assert payload["store"] is False
        assert payload["temperature"] == 0.7
        response = self.responses.popleft()
        if not payload.get("stream", False):
            return httpx2.Response(200, json=response)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_events(response),
        )


class FixtureKnowledgeBase:
    vector_store_id = "vs_docs"

    def __init__(self, *, index_status="indexed", upload_error=False):
        self.queries = []
        self.uploads = []
        self.indexed = []
        self.index_status = index_status
        self.upload_error = upload_error

    async def search(self, query):
        self.queries.append(query)
        return [KnowledgeResult("file_policy", "policy.md", "Retain for 30 days.")]

    async def upload(self, filename, content):
        self.uploads.append((filename, content))
        if self.upload_error:
            raise httpx2.ReadError("connection lost")
        return "file_saved"

    async def index(self, file_id):
        self.indexed.append(file_id)
        return self.index_status


@tool("web_search", response_format="content_and_artifact")
async def fixture_web_search(query: str):
    """Search current public documentation."""
    return f"LangGraph docs: {SOURCE_URL}", {SOURCE_URL: "LangGraph interrupts"}


@asynccontextmanager
async def graph_client(
    checkpointer,
    provider,
    *,
    knowledge=None,
    store=None,
    files_handler=None,
):
    store = store or InMemoryStore()

    async def unexpected_file_request(request):
        raise AssertionError(f"Unexpected Files API request: {request.url}")

    async with (
        httpx2.AsyncClient(
            transport=httpx2.MockTransport(provider.handle)
        ) as upstream_http,
        httpx2.AsyncClient(
            transport=httpx2.MockTransport(files_handler or unexpected_file_request)
        ) as files_http,
        AsyncOpenAI(
            api_key="test",
            base_url="http://files.test/v1",
            http_client=files_http,
            max_retries=0,
        ) as files_client,
    ):
        graph = create_advanced_graph(
            model=create_model(upstream_http),
            knowledge=knowledge,
            files=files_client,
            checkpointer=checkpointer,
            store=store,
            web_search_tool=fixture_web_search,
        )
        config = create_advanced_graph_config(
            lambda: graph,
            InMemoryRunCoordinator(),
        )
        app = (
            LanggraphOpenaiServe(
                graphs=GraphRegistry(registry={"advanced-graph": config})
            )
            .bind_openai_api()
            .app
        )
        async with (
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as transport,
            AsyncOpenAI(
                api_key="test",
                base_url="http://test/v1",
                http_client=transport,
                max_retries=0,
            ) as client,
        ):
            yield client


def resume_input(response, decision):
    return [
        {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": decision,
        }
        for item in response.output
        if item.type == "function_call"
    ]


async def test_research_uses_existing_web_contract_and_internal_vector_search(
    sqlite_checkpointer,
):
    calls = [
        {
            "type": "function_call",
            "id": "fc_web",
            "call_id": "call_web",
            "name": "web_search",
            "arguments": '{"query":"LangGraph interrupts"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": "fc_knowledge",
            "call_id": "call_knowledge",
            "name": "knowledge_search",
            "arguments": '{"query":"retention policy"}',
            "status": "completed",
        },
    ]
    answer = f"See [LangGraph]({SOURCE_URL}) and policy.md (file_policy) [K1]."
    provider = ModelProvider(
        intent_response("research"),
        model_response(calls=calls),
        model_response(answer),
    )
    knowledge = FixtureKnowledgeBase()
    async with graph_client(
        sqlite_checkpointer, provider, knowledge=knowledge
    ) as client:
        stream = await client.responses.create(
            model="advanced-graph",
            input="Compare interrupt and retention rules",
            tools=WEB_TOOL,
            stream=True,
        )
        events = [event async for event in stream]

    final = events[-1].response
    assert knowledge.queries == ["retention policy"]
    assert [item.type for item in final.output].count("web_search_call") == 1
    assert "knowledge_search" not in final.model_dump_json()
    message = next(
        item
        for item in final.output
        if item.type == "message" and item.phase == "final_answer"
    )
    assert message.content[0].text == answer
    assert message.content[0].annotations[0].url == SOURCE_URL
    assert any(
        item.type == "message" and item.phase == "commentary" for item in final.output
    )


async def test_research_and_save_runs_both_subgraphs(sqlite_checkpointer):
    web_call = {
        "type": "function_call",
        "id": "fc_web",
        "call_id": "call_web",
        "name": "web_search",
        "arguments": '{"query":"LangGraph interrupts"}',
        "status": "completed",
    }
    provider = ModelProvider(
        intent_response("research_and_save"),
        model_response(calls=[web_call]),
        model_response("# Durable finding"),
        model_response("The reviewed finding is now searchable."),
    )
    knowledge = FixtureKnowledgeBase()
    async with graph_client(
        sqlite_checkpointer, provider, knowledge=knowledge
    ) as client:
        paused = await client.responses.create(
            model="advanced-graph",
            input="Research LangGraph interrupts and save the result for later.",
            tools=WEB_TOOL,
        )
        completed = await client.responses.create(
            model="advanced-graph",
            previous_response_id=paused.id,
            input=resume_input(paused, "approve"),
            tools=WEB_TOOL,
        )

    review_call = next(
        item
        for item in paused.output
        if item.type == "function_call" and item.name == "lgos_interrupt"
    )
    reviewed = json.loads(review_call.arguments)
    assert reviewed["content"] == "# Durable finding"
    assert knowledge.uploads == [(reviewed["filename"], b"# Durable finding")]
    assert knowledge.indexed == ["file_saved"]
    assert completed.output_text == "The reviewed finding is now searchable."


async def test_file_ids_are_resolved_through_the_compatible_files_api(
    sqlite_checkpointer,
):
    file_id = "file_attachment"
    filename = "brief.txt"
    content = b"The attachment marker is FILE_INPUT_E2E."
    file_requests = []

    async def handle_files(request):
        file_requests.append(request)
        if request.url.path.endswith("/content"):
            return httpx2.Response(
                200,
                content=content,
                headers={"content-type": "text/plain"},
            )
        return httpx2.Response(
            200,
            json={
                "id": file_id,
                "object": "file",
                "bytes": len(content),
                "created_at": 1,
                "filename": filename,
                "purpose": "user_data",
                "status": "processed",
            },
        )

    provider = ModelProvider(
        intent_response("chat"),
        model_response("FILE_INPUT_E2E"),
    )
    async with graph_client(
        sqlite_checkpointer,
        provider,
        files_handler=handle_files,
    ) as client:
        response = await client.responses.create(
            model="advanced-graph",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read the attachment."},
                        {"type": "input_file", "file_id": file_id},
                    ],
                }
            ],
            tool_choice="none",
        )

    assert response.output_text.endswith("FILE_INPUT_E2E")
    assert [request.url.path for request in file_requests] == [
        f"/v1/files/{file_id}",
        f"/v1/files/{file_id}/content",
    ]
    router_input = next(
        item for item in provider.requests[0]["input"] if item["role"] == "user"
    )
    assert isinstance(router_input["content"], str)
    assert "attached one or more files" in router_input["content"]
    user_input = next(
        item for item in provider.requests[1]["input"] if item["role"] == "user"
    )
    file_input = next(
        item for item in user_input["content"] if item["type"] == "input_file"
    )
    assert file_input == {
        "type": "input_file",
        "filename": filename,
        "file_data": "data:text/plain;base64," + b64encode(content).decode("ascii"),
    }


async def test_client_function_remains_client_owned(sqlite_checkpointer):
    call = {
        "type": "function_call",
        "id": "fc_order",
        "call_id": "call_order",
        "name": "lookup_order",
        "arguments": '{"order_id":"123"}',
        "status": "completed",
    }
    provider = ModelProvider(model_response(calls=[call]))
    knowledge = FixtureKnowledgeBase()
    async with graph_client(
        sqlite_checkpointer, provider, knowledge=knowledge
    ) as client:
        response = await client.responses.create(
            model="advanced-graph",
            input="Look up order 123",
            tools=[
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
                }
            ],
            tool_choice={"type": "function", "name": "lookup_order"},
            parallel_tool_calls=False,
        )

    assert knowledge.queries == []
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request["tools"][0]["name"] == "lookup_order"
    assert request["tools"][0]["strict"] is True
    assert request["tool_choice"] == {"type": "function", "name": "lookup_order"}
    assert request["parallel_tool_calls"] is False
    output_call = next(item for item in response.output if item.type == "function_call")
    assert output_call.call_id == "call_order"
    assert output_call.name == "lookup_order"
    assert output_call.arguments == '{"order_id":"123"}'


async def test_plain_chat_stream_has_no_synthetic_status(sqlite_checkpointer):
    provider = ModelProvider(
        intent_response("chat"),
        model_response("Hello from the assistant."),
    )
    async with graph_client(sqlite_checkpointer, provider) as client:
        stream = await client.responses.create(
            model="advanced-graph",
            input="Hello",
            stream=True,
        )
        events = [event async for event in stream]

    final = events[-1].response
    messages = [item for item in final.output if item.type == "message"]
    assert [message.phase for message in messages] == ["final_answer"]
    assert final.output_text == "Hello from the assistant."


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_exact_note_review_survives_restart(tmp_path, decision):
    database = str(tmp_path / "approval.sqlite")
    store = InMemoryStore()
    knowledge = FixtureKnowledgeBase()
    first_provider = ModelProvider(
        intent_response("save"),
        model_response("# Exact note"),
    )
    async with (
        AsyncSqliteSaver.from_conn_string(database) as checkpointer,
        graph_client(
            checkpointer,
            first_provider,
            knowledge=knowledge,
            store=store,
        ) as client,
    ):
        paused = await client.responses.create(
            model="advanced-graph", input="Remember this in a note: exact content"
        )
    reviewed = json.loads(paused.output[0].arguments)
    assert reviewed["content"] == "# Exact note"
    assert not knowledge.uploads

    second_provider = ModelProvider(model_response("Workflow finished."))
    async with (
        AsyncSqliteSaver.from_conn_string(database) as checkpointer,
        graph_client(
            checkpointer,
            second_provider,
            knowledge=knowledge,
            store=store,
        ) as client,
    ):
        completed = await client.responses.create(
            model="advanced-graph",
            previous_response_id=paused.id,
            input=resume_input(paused, decision),
        )
        with pytest.raises(ConflictError):
            await client.responses.create(
                model="advanced-graph",
                previous_response_id=paused.id,
                input=resume_input(paused, decision),
            )
    assert completed.status == "completed"
    assert len(knowledge.uploads) == (1 if decision == "approve" else 0)
    expected_workflow = (
        "indexing status indexed" if decision == "approve" else "nothing was saved"
    )
    assert expected_workflow in json.dumps(second_provider.requests[0])
    if decision == "approve":
        assert knowledge.uploads[0][1] == b"# Exact note"
        receipts = await store.asearch(("advanced-graph", "notes", "vs_docs"))
        assert receipts[0].value["status"] == "indexed"
        assert "content" not in receipts[0].value


async def test_review_feedback_redrafts_before_upload(sqlite_checkpointer):
    provider = ModelProvider(
        intent_response("save"),
        model_response("# First draft"),
        model_response("# Revised draft"),
        model_response("The revised note was saved."),
    )
    knowledge = FixtureKnowledgeBase()
    async with graph_client(
        sqlite_checkpointer, provider, knowledge=knowledge
    ) as client:
        first_review = await client.responses.create(
            model="advanced-graph", input="Remember this as a note."
        )
        second_review = await client.responses.create(
            model="advanced-graph",
            previous_response_id=first_review.id,
            input=resume_input(first_review, "Add a clearer title."),
        )
        completed = await client.responses.create(
            model="advanced-graph",
            previous_response_id=second_review.id,
            input=resume_input(second_review, "approve"),
        )

    first_note = json.loads(first_review.output[0].arguments)
    revised_note = json.loads(second_review.output[0].arguments)
    assert first_note["content"] == "# First draft"
    assert revised_note["content"] == "# Revised draft"
    assert revised_note["filename"] == first_note["filename"]
    assert knowledge.uploads == [(first_note["filename"], b"# Revised draft")]
    assert completed.output_text == "The revised note was saved."


@pytest.mark.parametrize("outcome", ["refusal", "incomplete"])
async def test_private_model_outcomes_are_preserved(sqlite_checkpointer, outcome):
    provider = ModelProvider(
        model_response(
            "private partial text",
            refusal="I cannot help." if outcome == "refusal" else None,
            incomplete=outcome == "incomplete",
        )
    )
    async with graph_client(
        sqlite_checkpointer, provider, knowledge=FixtureKnowledgeBase()
    ) as client:
        result = await client.responses.create(model="advanced-graph", input="request")
    assert len(provider.requests) == 1
    if outcome == "refusal":
        assert result.output[-1].content[0].refusal == "I cannot help."
    else:
        assert result.status == "incomplete"
        assert "private partial text" not in result.model_dump_json()


async def test_uncertain_upload_is_not_repeated(sqlite_checkpointer):
    knowledge = FixtureKnowledgeBase(upload_error=True)
    store = InMemoryStore()
    provider = ModelProvider(intent_response("save"), model_response("Note"))
    async with graph_client(
        sqlite_checkpointer, provider, knowledge=knowledge, store=store
    ) as client:
        paused = await client.responses.create(
            model="advanced-graph", input="Save this as a note."
        )
        with pytest.raises(InternalServerError):
            await client.responses.create(
                model="advanced-graph",
                previous_response_id=paused.id,
                input=resume_input(paused, "approve"),
            )
    assert len(knowledge.uploads) == 1
    receipts = await store.asearch(("advanced-graph", "notes", "vs_docs"))
    assert receipts[0].value["status"] == "upload_pending"


async def test_unchanged_public_contract_and_unavailable_storage(
    sqlite_checkpointer,
):
    provider = ModelProvider(
        intent_response("save"),
        model_response("Persistent storage is unavailable, so nothing was saved."),
    )
    async with graph_client(sqlite_checkpointer, provider) as client:
        with pytest.raises(BadRequestError):
            await client.chat.completions.create(
                model="advanced-graph", messages=[{"role": "user", "content": "Hi"}]
            )
        with pytest.raises(BadRequestError):
            await client.responses.create(
                model="advanced-graph",
                input="Search",
                tools=[{"type": "file_search", "vector_store_ids": ["vs_docs"]}],
            )
        response = await client.responses.create(
            model="advanced-graph", input="Remember this for later."
        )
    assert response.output_text == (
        "Persistent storage is unavailable, so nothing was saved."
    )
    assert len(provider.requests) == 2
    assert "Persistent note storage is unavailable" in json.dumps(provider.requests[-1])


async def test_compatible_storage_adapter_uses_configured_endpoint():
    requests = []

    async def handle(request):
        requests.append(request)
        if request.url.path.endswith("/search"):
            return httpx2.Response(
                200,
                json={
                    "object": "vector_store.search_results.page",
                    "data": [
                        {
                            "file_id": "file_policy",
                            "filename": "policy.md",
                            "score": 0.9,
                            "attributes": {},
                            "content": [{"type": "text", "text": "Policy text"}],
                        }
                    ],
                    "has_more": False,
                    "next_page": None,
                    "search_query": ["policy"],
                },
            )
        if request.url.path == "/v1/files":
            return httpx2.Response(
                200,
                json={
                    "id": "file_saved",
                    "object": "file",
                    "filename": "note.md",
                    "purpose": "user_data",
                    "bytes": 4,
                    "created_at": 1,
                    "status": "processed",
                },
            )
        if request.method == "GET" and len(requests) == 3:
            return httpx2.Response(
                404,
                json={"error": {"message": "missing", "type": "invalid_request_error"}},
            )
        return httpx2.Response(
            200,
            json={
                "id": "file_saved",
                "object": "vector_store.file",
                "vector_store_id": "vs_docs",
                "created_at": 1,
                "usage_bytes": 4,
                "status": "completed",
                "last_error": None,
            },
        )

    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http,
        AsyncOpenAI(
            api_key="compatible",
            base_url="https://storage.test/v1",
            http_client=http,
            max_retries=0,
        ) as client,
    ):
        knowledge = OpenAICompatibleKnowledgeBase(client, "vs_docs")
        assert (await knowledge.search("policy"))[0].text == "Policy text"
        file_id = await knowledge.upload("note.md", b"Note")
        assert await knowledge.index(file_id) == "indexed"
    assert {request.url.host for request in requests} == {"storage.test"}
    assert [request.url.path for request in requests] == [
        "/v1/vector_stores/vs_docs/search",
        "/v1/files",
        "/v1/vector_stores/vs_docs/files/file_saved",
        "/v1/vector_stores/vs_docs/files",
        "/v1/vector_stores/vs_docs/files/file_saved",
    ]
    assert b"user_data" in requests[1].content


@pytest.mark.parametrize("cancel", [False, True])
async def test_answer_stream_is_live_and_cancellable(sqlite_checkpointer, cancel):
    release = Event()
    closed = Event()
    completed = Event()
    request_count = 0

    class GatedStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            frames = response_events(model_response("Hello world")).split(b"\n\n")
            for frame in frames:
                if b'"delta": "Hello world"' in frame:
                    yield frame.replace(b"Hello world", b"Hello") + b"\n\n"
                    await release.wait()
                    yield frame.replace(b"Hello world", b" world") + b"\n\n"
                    continue
                yield frame + b"\n\n"
            completed.set()

        async def aclose(self):
            closed.set()

    async def handle(request):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            assert not json.loads(request.content)["stream"]
            return httpx2.Response(200, json=intent_response("chat"))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=GatedStream(),
        )

    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http,
        AsyncOpenAI(
            api_key="test",
            base_url="http://files.test/v1",
            max_retries=0,
        ) as files,
    ):
        graph = create_advanced_graph(
            model=create_model(http),
            knowledge=None,
            files=files,
            checkpointer=sqlite_checkpointer,
            store=InMemoryStore(),
            web_search_tool=fixture_web_search,
        )
        registry = GraphRegistry(
            registry={
                "advanced-graph": create_advanced_graph_config(
                    lambda: graph,
                    InMemoryRunCoordinator(),
                )
            }
        )
        request = ResponseCreateRequest(
            model="advanced-graph", input="Hello", stream=True
        )
        decoded, messages, _ = decode_responses_request(request, set())
        run = await prepare_run(decoded, messages, registry)
        events = []
        with fail_after(5):
            async with aclosing(stream_response(request, run)) as stream:
                async for frame in stream:
                    event = json.loads(frame.split("data: ", 1)[1])
                    events.append(event)
                    if (
                        event["type"] == "response.output_text.delta"
                        and event["delta"] == "Hello"
                    ):
                        assert not completed.is_set()
                        if cancel:
                            break
                        release.set()
            await closed.wait()
    assert request_count == 2
    assert completed.is_set() is not cancel
    if not cancel:
        assert events[-1]["type"] == "response.completed"
        assert (
            events[-1]["response"]["output"][-1]["content"][0]["text"] == "Hello world"
        )
