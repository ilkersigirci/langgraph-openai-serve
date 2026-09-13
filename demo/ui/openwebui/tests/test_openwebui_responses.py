"""Responses-only Open WebUI Function behavior."""

import json
import sys
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseWebSearchCallCompletedEvent,
)
from openai.types.responses.parsed_response import ParsedResponseFunctionToolCall
from openai.types.responses.response_output_text import AnnotationURLCitation

from lgos_openwebui.bundle import bundle_function
from lgos_openwebui.functions.generic import files as generic_files
from lgos_openwebui.functions.generic import pipe as generic_pipe
from lgos_openwebui.functions.generic.interrupts import (
    _ask_user_to_resume,
    _interrupts_to_ask_user,
)
from lgos_openwebui.functions.generic.responses import _responses_input
from lgos_openwebui.functions.uservalves_simple import Filter

pytestmark = pytest.mark.usefixtures("gateway_environment")

MODEL_ID = "interruptible-approval"
QUALIFIED_MODEL_ID = f"generic.{MODEL_ID}"
RESPONSE_ID = "resp_lg_725c277af6d54c5295eb8c09e91f7a7c_" + "b" * 32


def response(*output: object) -> Response:
    return Response.model_construct(
        id=RESPONSE_ID,
        status="completed",
        output=list(output),
    )


def final_response(text: str) -> Response:
    return response(
        ResponseOutputMessage(
            id="msg_final",
            content=[
                ResponseOutputText(
                    annotations=[],
                    logprobs=[],
                    text=text,
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
            phase="final_answer",
        )
    )


def function_call(name: str, arguments: dict[str, object]) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{name}",
        call_id=f"call_{name}",
        name=name,
        arguments=json.dumps(arguments, separators=(",", ":")),
        status="completed",
        type="function_call",
    )


def interrupt_call() -> ResponseFunctionToolCall:
    return function_call(
        "lgos_interrupt",
        {
            "question": "Approve refund?",
            "choices": ["approve", "reject"],
            "allow_other": False,
        },
    )


def body(*, stream: bool) -> dict[str, object]:
    return {
        "model": QUALIFIED_MODEL_ID,
        "messages": [{"role": "user", "content": "Refund ORDER-123"}],
        "stream": stream,
    }


async def collect(
    value: Awaitable[AsyncIterator[str | dict[str, Any]] | str | dict[str, Any]],
) -> list[str | dict[str, Any]]:
    result = await value
    if isinstance(result, (str, dict)):
        return [result]
    return [item async for item in result]


class FakeResponseStream:
    def __init__(self, events: Sequence[object], completed: Response) -> None:
        self.events = events
        self.completed = completed

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self.events:
            yield event

    async def get_final_response(self) -> Response:
        return self.completed


class FakeClient:
    def __init__(self, **responses: object) -> None:
        self.responses = SimpleNamespace(**responses)

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


def install_client(monkeypatch: pytest.MonkeyPatch, **responses: object) -> None:
    monkeypatch.setattr(generic_pipe, "_client", lambda **_: FakeClient(**responses))


@pytest.fixture
def bundled_generic(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    source = bundle_function(Path(generic_pipe.__file__).parent)
    module = ModuleType("bundled_generic")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "<generic>", "exec"), module.__dict__)
    return module


async def test_pipe_lists_native_litellm_model_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = {"model_name": "research/graph", "model_info": {"lgos": {}}}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/model/info"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert "x-model-provider" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    deployment,
                    deployment,
                    {"model_name": "plain", "model_info": {"lgos": {}}},
                    {"model_name": "gpt-5", "model_info": {}},
                ]
            },
        )

    @asynccontextmanager
    async def catalog_client(**kwargs: Any) -> AsyncIterator[AsyncOpenAI]:
        async with AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            yield client

    monkeypatch.setattr(generic_pipe, "_client", catalog_client)
    pipe = generic_pipe.Pipe()
    pipe.valves.OPENAI_GATEWAY_API_KEY = "test-key"
    models = await pipe.pipes()
    assert models == [
        {"id": "research/graph", "name": "Generic / research/graph"},
        {"id": "plain", "name": "Generic / plain"},
    ]


async def test_pipe_uses_bifrost_aggregate_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_urls = []

    @asynccontextmanager
    async def catalog_client(*, base_url: str, **_: object) -> AsyncIterator[object]:
        catalog_urls.append(base_url)
        yield SimpleNamespace(
            models=SimpleNamespace(
                list=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[
                            SimpleNamespace(
                                id="lgos-a/simple-graph",
                                owned_by="langgraph-openai-serve",
                            ),
                            SimpleNamespace(
                                id="lgos-b/simple-graph",
                                owned_by="langgraph-openai-serve",
                            ),
                        ]
                    )
                )
            )
        )

    monkeypatch.setattr(generic_pipe, "_client", catalog_client)
    pipe = generic_pipe.Pipe()
    pipe.valves.OPENAI_GATEWAY_TYPE = "bifrost"
    pipe.valves.OPENAI_GATEWAY_BASE_URL = "https://bifrost.example"

    models = await pipe.pipes()

    assert catalog_urls == ["https://bifrost.example/v1"]
    assert models == [
        {"id": "lgos-a/simple-graph", "name": "Generic / lgos-a/simple-graph"},
        {"id": "lgos-b/simple-graph", "name": "Generic / lgos-b/simple-graph"},
    ]


async def test_deployed_bundle_runs_responses_inference(
    bundled_generic: ModuleType,
) -> None:
    create = AsyncMock(return_value=final_response("Bundle answer."))
    bundled_generic._client = lambda **_: FakeClient(create=create)

    result = await bundled_generic.Pipe().pipe(body(stream=False))

    assert result == "Bundle answer."
    request = create.await_args.kwargs
    assert request["input"] == [{"role": "user", "content": "Refund ORDER-123"}]
    assert request["store"] is False


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("runtime_settings", [{}, {"audience": "expert"}])
async def test_bundle_maps_server_controls_without_forwarding_openwebui_tools(
    bundled_generic, streaming, runtime_settings
):
    openwebui_tools = [
        {
            "type": "function",
            "function": {
                "name": "client_tool",
                "description": "An OpenWebUI client tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    completed = final_response("It is noon.")
    completed.output[:0] = [
        ResponseCustomToolCall.model_validate(
            {
                "type": "custom_tool_call",
                "id": "ctc_clock",
                "call_id": "call_clock",
                "name": "lgos_current_time",
                "input": "UTC",
                "status": "completed",
            }
        ),
        ResponseCustomToolCallOutputItem(
            type="custom_tool_call_output",
            id="ctco_clock",
            call_id="call_clock",
            status="completed",
            output="Noon",
        ),
    ]
    requests = []

    async def create(**request):
        requests.append(request)
        return completed

    @asynccontextmanager
    async def stream(**request):
        requests.append(request)
        yield FakeResponseStream([], completed)

    bundled_generic._client = lambda **_: FakeClient(create=create, stream=stream)
    result = await collect(
        bundled_generic.Pipe().pipe(
            {
                **body(stream=streaming),
                "model": "generic.lgos-a/server-tool",
                "tools": openwebui_tools,
            },
            __metadata__={
                "chat_id": "thread-123",
                "chat_variables": {
                    **runtime_settings,
                    "lgos_current_time": True,
                    "web_search": True,
                },
            },
        )
    )
    assert len(requests) == 1
    assert requests[0]["tools"] == [
        {"type": "custom", "name": "lgos_current_time"},
        {"type": "web_search"},
    ]
    expected_metadata = {"conversation_id": "thread-123"}
    if runtime_settings:
        expected_metadata["lgos_settings"] = '{"audience":"expert"}'
    assert requests[0]["metadata"] == expected_metadata
    assert (
        result[0]["choices"][0]["delta"]["content"] if streaming else result[0]
    ) == "It is noon."


async def test_deployed_bundle_runs_non_streaming_interrupt(
    bundled_generic: ModuleType,
) -> None:
    create = AsyncMock(return_value=response(interrupt_call()))
    bundled_generic._client = lambda **_: FakeClient(create=create)

    result = await bundled_generic.Pipe().pipe(body(stream=False))

    assert (
        result["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "ask_user"
    )
    assert result["output"][0]["name"] == "ask_user"
    assert result["output"][0]["status"] == "pending"


@pytest.mark.parametrize("deltas", [False, True])
async def test_bundled_stream_keeps_sse_looking_text_as_content(
    bundled_generic: ModuleType, deltas: bool
) -> None:
    chunks = ["data: [DONE]", '\n\ndata: {"error":"example"}', "\nStill text."]
    answer = "".join(chunks)
    events = (
        [
            SimpleNamespace(
                type="response.output_text.delta", output_index=0, delta=text
            )
            for text in chunks
        ]
        if deltas
        else []
    )

    @asynccontextmanager
    async def scripted_stream(**_: object) -> AsyncIterator[FakeResponseStream]:
        yield FakeResponseStream(events, final_response(answer))

    bundled_generic._client = lambda **_: FakeClient(stream=scripted_stream)

    output = await collect(bundled_generic.Pipe().pipe(body(stream=True)))

    # The host JSON-encodes objects; raw strings beginning with data: bypass it.
    decoded = [ChatCompletionChunk.model_validate(chunk) for chunk in output]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in decoded) == answer
    assert all(chunk.choices[0].finish_reason is None for chunk in decoded)


async def test_non_streaming_request_uses_responses_and_final_answer_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=final_response("Approved."))
    install_client(monkeypatch, create=create)
    pipe = generic_pipe.Pipe()

    result = await pipe.pipe(
        body(stream=False),
        __metadata__={
            "chat_id": "thread-123",
            "chat_variables": {
                "audience": "expert",
                "lgos_current_time": False,
                "web_search": True,
            },
        },
        __user__={"id": "user-123"},
    )

    assert result == "Approved."
    request = create.await_args.kwargs
    assert request["model"] == "interruptible-approval"
    assert "extra_headers" not in request
    assert request["input"] == [{"role": "user", "content": "Refund ORDER-123"}]
    assert request["store"] is False
    assert request["user"] == "user-123"
    assert request["metadata"] == {
        "conversation_id": "thread-123",
        "lgos_settings": '{"audience":"expert"}',
    }
    assert request["tools"][0]["name"] == "display_file"


@pytest.mark.parametrize(
    "settings",
    [Filter.UserValves(), Filter.UserValves(use_history=True, audience="beginner")],
)
async def test_uservalves_reach_responses_through_shared_pipe(
    monkeypatch: pytest.MonkeyPatch, settings: Filter.UserValves
) -> None:
    create = AsyncMock(return_value=final_response("Hello."))
    install_client(monkeypatch, create=create)
    metadata = {
        "chat_id": "thread-123",
        "chat_variables": {"audience": "expert", "stale_setting": True},
    }
    request_body = {
        "model": "lgos.uservalves_simple",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    filtered = await Filter().inlet(request_body, {"valves": settings}, metadata)
    # Open WebUI resolves the Workspace Model to its manifold base before Pipe.
    filtered["model"] = "generic.lgos-a/simple-graph"

    result = await generic_pipe.Pipe().pipe(
        filtered, __metadata__=metadata, __user__={"id": "user-123"}
    )

    assert result == "Hello."
    request = create.await_args.kwargs
    assert request["model"] == "lgos-a/simple-graph"
    assert request["metadata"]["conversation_id"] == "thread-123"
    assert json.loads(request["metadata"]["lgos_settings"]) == {
        "use_history": settings.use_history,
        "audience": settings.audience,
    }
    assert request["input"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("gateway_type", "base_path", "expected_model", "extra_headers"),
    [
        (
            "bifrost",
            "/openai/v1",
            "interruptible-approval",
            {"x-model-provider": "lgos-a"},
        ),
        ("litellm", "/v1", "lgos-a/interruptible-approval", None),
    ],
)
async def test_request_uses_a_native_responses_route(
    monkeypatch: pytest.MonkeyPatch,
    gateway_type: str,
    base_path: str,
    expected_model: str,
    extra_headers: dict[str, str] | None,
    stream: bool,
) -> None:
    base_urls = []

    @asynccontextmanager
    async def client(*, base_url: str, **_: object) -> AsyncIterator[object]:
        base_urls.append(base_url)
        yield FakeClient(create=create, stream=stream_request)

    completed = final_response("Approved.")
    create = AsyncMock(return_value=completed)

    @asynccontextmanager
    async def response_stream(**_: object) -> AsyncIterator[FakeResponseStream]:
        yield FakeResponseStream([], completed)

    stream_request = Mock(side_effect=response_stream)
    monkeypatch.setattr(generic_pipe, "_client", client)
    pipe = generic_pipe.Pipe()
    pipe.valves.OPENAI_GATEWAY_TYPE = gateway_type
    pipe.valves.OPENAI_GATEWAY_BASE_URL = "https://gateway.example"

    result = await collect(
        pipe.pipe(
            {
                **body(stream=stream),
                "model": "generic.lgos-a/interruptible-approval",
            },
            __metadata__={"chat_id": "thread-123"},
            __user__={"id": "user-123"},
        )
    )

    if stream:
        chunk = ChatCompletionChunk.model_validate(result[0])
        assert chunk.choices[0].delta.content == "Approved."
        request = stream_request.call_args.kwargs
    else:
        assert result == ["Approved."]
        request = create.await_args.kwargs
    assert base_urls == [f"https://gateway.example{base_path}"]
    assert request["model"] == expected_model
    assert request.get("extra_headers") == extra_headers


@pytest.mark.parametrize("phase", [None, "final_answer"])
async def test_stream_uses_sdk_final_response_and_excludes_commentary(
    monkeypatch: pytest.MonkeyPatch,
    phase: str | None,
) -> None:
    commentary = SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(type="message", phase="commentary"),
    )
    commentary_done = SimpleNamespace(
        type="response.output_text.done",
        output_index=0,
        text="Checking policy",
    )
    final_added = SimpleNamespace(
        type="response.output_item.added",
        output_index=1,
        item=SimpleNamespace(type="message", phase=phase),
    )
    final_delta = SimpleNamespace(
        type="response.output_text.delta",
        output_index=1,
        delta="Approved.",
    )
    stream = FakeResponseStream(
        [
            commentary,
            commentary_done,
            ResponseWebSearchCallCompletedEvent(
                type="response.web_search_call.completed",
                item_id="ws_123",
                output_index=1,
                sequence_number=2,
            ),
            final_added,
            final_delta,
        ],
        final_response("Approved."),
    )

    @asynccontextmanager
    async def scripted_stream(**_: object) -> AsyncIterator[FakeResponseStream]:
        yield stream

    emit = AsyncMock()
    install_client(monkeypatch, stream=scripted_stream)
    chunks = await collect(
        generic_pipe.Pipe().pipe(
            body(stream=True),
            __event_emitter__=emit,
        )
    )

    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "Approved."
    assert [call.args[0] for call in emit.await_args_list] == [
        {
            "type": "status",
            "data": {"description": "Checking policy", "done": False},
        },
        {
            "type": "status",
            "data": {"description": "Web search completed.", "done": True},
        },
        {
            "type": "status",
            "data": {"description": "Checking policy", "done": True},
        },
    ]


@pytest.mark.parametrize(
    ("streaming", "send_delta"),
    [(False, False), (True, False), (True, True)],
)
async def test_refusal_is_visible_in_both_response_modes(
    monkeypatch, streaming, send_delta
):
    completed = final_response("")
    completed.output[0].content = [
        ResponseOutputRefusal(type="refusal", refusal="Cannot answer this request.")
    ]
    events = (
        [
            SimpleNamespace(
                type="response.refusal.delta",
                output_index=0,
                delta="Cannot answer this request.",
            )
        ]
        if send_delta
        else []
    )

    @asynccontextmanager
    async def scripted_stream(**_):
        yield FakeResponseStream(events, completed)

    install_client(
        monkeypatch, stream=scripted_stream, create=AsyncMock(return_value=completed)
    )
    chunks = await collect(generic_pipe.Pipe().pipe(body(stream=streaming)))
    assert len(chunks) == 1
    assert (
        chunks[0]["choices"][0]["delta"]["content"] if streaming else chunks[0]
    ) == "Cannot answer this request."


async def test_failed_stream_closes_running_status_and_does_not_execute_tools(
    monkeypatch,
):
    from openai.types.responses.response import IncompleteDetails

    completed = response(function_call("display_file", {})).model_copy(
        update={
            "status": "incomplete",
            "incomplete_details": IncompleteDetails(reason="max_output_tokens"),
        }
    )
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(type="message", phase="commentary"),
        ),
        SimpleNamespace(
            type="response.output_text.done", output_index=0, text="Making chart"
        ),
        SimpleNamespace(type="response.incomplete", response=completed),
    ]

    @asynccontextmanager
    async def scripted_stream(**_):
        stream = FakeResponseStream(events, completed)
        stream.get_final_response = AsyncMock(
            side_effect=RuntimeError("No completed response")
        )
        yield stream

    install_client(monkeypatch, stream=scripted_stream)
    display = AsyncMock()
    monkeypatch.setattr(generic_pipe, "_handle_display_file", display)
    emit = AsyncMock()
    chunks = await collect(
        generic_pipe.Pipe().pipe(body(stream=True), __event_emitter__=emit)
    )
    assert "max_output_tokens" in chunks[0]["error"]["detail"]
    assert [call.args[0]["data"] for call in emit.await_args_list] == [
        {"description": "Making chart", "done": False},
        {"description": "Stopped: Making chart", "done": True},
    ]
    display.assert_not_awaited()


@pytest.mark.parametrize("streaming", [False, True])
async def test_response_maps_final_answer_annotations_to_persistent_sources(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    text = "🌍 Café source"
    cited_text = "source"
    annotation = AnnotationURLCitation(
        type="url_citation",
        url="https://example.com/source",
        title="Example source",
        start_index=text.index(cited_text),
        end_index=len(text) - 1,
    )
    final_added = SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(type="message", phase="final_answer"),
    )
    final_delta = SimpleNamespace(
        type="response.output_text.delta",
        output_index=0,
        delta=text,
    )
    completed = final_response(text)
    completed.output[0].content[0].annotations = [annotation]
    stream = FakeResponseStream([final_added, final_delta], completed)

    @asynccontextmanager
    async def scripted_stream(**_: object) -> AsyncIterator[FakeResponseStream]:
        yield stream

    emit = AsyncMock()
    install_client(
        monkeypatch, stream=scripted_stream, create=AsyncMock(return_value=completed)
    )

    chunks = await collect(
        generic_pipe.Pipe().pipe(
            body(stream=streaming),
            __event_emitter__=emit,
        )
    )

    assert (
        chunks[0]["choices"][0]["delta"]["content"] if streaming else chunks[0]
    ) == text
    emit.assert_awaited_once_with(
        {
            "type": "source",
            "data": {
                "source": {
                    "name": "Example source",
                    "url": "https://example.com/source",
                },
                "document": [cited_text],
                "metadata": [
                    {
                        "source": "Example source",
                        "name": "Example source",
                        "url": "https://example.com/source",
                    }
                ],
            },
        }
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("resume_interrupt", [False, True])
async def test_display_file_continuation_preserves_input_and_all_final_text(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    resume_interrupt: bool,
) -> None:
    call = function_call(
        "display_file",
        {
            "file_id": "file-chart",
            "filename": "chart.png",
            "media_type": "image/png",
            "title": "Revenue",
            "alt": "Q4 is highest.",
        },
    )
    server_call = ResponseCustomToolCall.model_validate(
        {
            "type": "custom_tool_call",
            "id": "ctc_clock",
            "call_id": "call_clock",
            "name": "lgos_current_time",
            "input": "UTC",
            "status": "completed",
        }
    )
    server_output = ResponseCustomToolCallOutputItem(
        type="custom_tool_call_output",
        id="ctco_clock",
        call_id=server_call.call_id,
        output="Noon",
        status="completed",
    )
    first = response(
        *final_response("Here is the chart. ").output,
        server_call,
        server_output,
        call,
    )
    expected_output = [
        item.model_dump(mode="json", exclude_none=True) for item in first.output
    ]
    if streaming:
        call = ParsedResponseFunctionToolCall(
            **expected_output[-1], parsed_arguments=json.loads(call.arguments)
        )
        first.output[-1] = call
    responses = iter([first, final_response("Chart ready.")])
    requests = []

    async def create(**request):
        requests.append(deepcopy(request))
        return next(responses)

    @asynccontextmanager
    async def stream(**request):
        completed = await create(**request)
        text_item = completed.output[0]
        yield FakeResponseStream(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    output_index=0,
                    item=text_item,
                ),
                SimpleNamespace(
                    type="response.output_text.delta",
                    output_index=0,
                    delta=text_item.content[0].text,
                ),
            ],
            completed,
        )

    output = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }
    handle = AsyncMock(return_value=output)
    install_client(monkeypatch, create=create, stream=stream)
    monkeypatch.setattr(generic_pipe, "_handle_display_file", handle)
    emitter = AsyncMock()
    request = object()
    request_body = body(stream=streaming)
    request_body["messages"] = [
        {"role": "system", "content": "Use the uploaded data."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Plot revenue"},
                {"type": "input_file", "file_id": "file-data"},
            ],
        },
    ]
    transcript = deepcopy(request_body["messages"])
    if resume_interrupt:
        ask_user = _interrupts_to_ask_user(RESPONSE_ID, [interrupt_call()])
        request_body["messages"].extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [ask_user]},
                {
                    "role": "tool",
                    "tool_call_id": ask_user["id"],
                    "content": json.dumps(
                        {
                            "status": "answered",
                            "answers": {
                                "resume_0": {"type": "option", "option_index": 0}
                            },
                        }
                    ),
                },
            ]
        )

    result = await collect(
        generic_pipe.Pipe().pipe(
            request_body, __event_emitter__=emitter, __request__=request
        )
    )

    text = (
        "".join(chunk["choices"][0]["delta"]["content"] for chunk in result)
        if streaming
        else result[0]
    )
    assert text == "Here is the chart. Chart ready."
    if resume_interrupt:
        assert requests[0]["previous_response_id"] == RESPONSE_ID
        assert requests[0]["input"][0]["type"] == "function_call_output"
    else:
        assert requests[0]["input"] == transcript
    assert "previous_response_id" not in requests[1]
    assert requests[1]["input"] == [
        *transcript,
        *expected_output,
        output,
    ]
    handle.assert_awaited_once()
    assert handle.await_args.args == (call, emitter, request)


@pytest.mark.parametrize("provider", ["lgos-files", "litellm_proxy"])
async def test_display_file_is_copied_to_authenticated_openwebui_storage(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    call = function_call(
        "display_file",
        {
            "file_id": "file-chart",
            "filename": "chart.png",
            "media_type": "image/png",
            "title": "Revenue",
            "alt": "Q4 is highest.",
        },
    )
    download = SimpleNamespace(aread=AsyncMock(return_value=b"png-bytes"))
    files_client = SimpleNamespace(
        files=SimpleNamespace(content=AsyncMock(return_value=download))
    )

    class FilesClientContext:
        async def __aenter__(self) -> object:
            return files_client

        async def __aexit__(self, *_: object) -> None:
            pass

    store = AsyncMock(return_value="openwebui-file")
    emit = AsyncMock()
    monkeypatch.setattr(generic_files, "_client", lambda **_: FilesClientContext())
    monkeypatch.setattr(generic_files, "_store_openwebui_file", store)

    output = await generic_files._handle_display_file(
        call,
        emit,
        object(),
        files_base_url="https://files.example/v1",
        api_key="test",
        timeout=10,
        provider=provider,
    )

    files_client.files.content.assert_awaited_once_with(
        "file-chart", extra_query={"provider": provider}
    )
    assert store.await_args.kwargs["content"] == b"png-bytes"
    emit.assert_awaited_once_with(
        {
            "type": "files",
            "data": {
                "files": [
                    {
                        "type": "image",
                        "url": "/api/v1/files/openwebui-file/content",
                        "name": "chart.png",
                    }
                ]
            },
        }
    )
    assert output["output"] == '{"displayed":true}'


@pytest.mark.parametrize(
    "invalid_content",
    [None, b"bad-json", b'{"data":{}}', b'{"data":[],"layout":[]}'],
    ids=["plotly", "invalid-json", "invalid-data", "invalid-layout"],
)
async def test_display_plotly_emits_a_persistent_interactive_embed(
    monkeypatch: pytest.MonkeyPatch,
    invalid_content: bytes | None,
) -> None:
    call = function_call(
        "display_file",
        {
            "file_id": "file-chart",
            "filename": "chart.plotly.json",
            "media_type": "application/vnd.plotly.v1+json",
            "title": "Revenue",
            "alt": "Q2 is highest.",
        },
    )
    chart = b'{"data":[{"type":"bar","x":["Q1","Q2"],"y":[120,180]}]}'
    download = SimpleNamespace(
        aread=AsyncMock(
            return_value=chart if invalid_content is None else invalid_content
        )
    )
    files_client = SimpleNamespace(
        files=SimpleNamespace(content=AsyncMock(return_value=download))
    )

    @asynccontextmanager
    async def client(**_: object):
        yield files_client

    emit = AsyncMock()
    monkeypatch.setattr(generic_files, "_client", client)
    kwargs = {
        "files_base_url": "https://files.example/v1",
        "api_key": "test",
        "timeout": 10,
        "provider": "lgos-files",
    }
    if invalid_content is not None:
        with pytest.raises(ValueError):
            await generic_files._handle_display_file(call, emit, object(), **kwargs)
        emit.assert_not_awaited()
        return

    output = await generic_files._handle_display_file(call, emit, object(), **kwargs)

    files_client.files.content.assert_awaited_once_with(
        "file-chart", extra_query={"provider": "lgos-files"}
    )
    emit.assert_awaited_once()
    event = emit.await_args.args[0]
    assert event["type"] == "embeds"
    (html,) = event["data"]["embeds"]
    assert "Plotly.newPlot" in html
    assert '"y":[120,180]' in html
    assert "https://cdn.plot.ly/plotly-" in html
    assert "files.example" not in html
    assert output == {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }


def test_plotly_labels_cannot_inject_html_into_the_embed() -> None:
    label = '</script><script>alert("injected")</script>'
    figure = {"data": [{"type": "bar", "x": [label], "y": [120]}]}

    html = generic_files._plotly_html(json.dumps(figure).encode())

    assert label not in html
    serialized = html.split("const figure = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(serialized) == figure
    assert html.count("<script") == 2


async def test_current_openwebui_attachment_becomes_responses_input_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"pdf-bytes")
    create = AsyncMock(return_value=SimpleNamespace(id="file-report"))
    files_client = SimpleNamespace(files=SimpleNamespace(create=create))

    class FilesClientContext:
        async def __aenter__(self) -> object:
            return files_client

        async def __aexit__(self, *_: object) -> None:
            pass

    monkeypatch.setattr(generic_files, "_client", lambda **_: FilesClientContext())

    messages = await generic_files._with_response_file_parts(
        [{"role": "user", "content": "Summarize it."}],
        [
            {
                "id": "owui-file",
                "type": "file",
                "file": {"path": str(path), "filename": "report.pdf"},
            }
        ],
        {
            "user_message": {
                "files": [
                    {
                        "id": "owui-file",
                        "type": "file",
                        "name": "report.pdf",
                        "content_type": "application/pdf",
                    }
                ]
            }
        },
        base_url="https://files.example/v1",
        api_key="test",
        timeout=10,
        provider="lgos-files",
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize it."},
                {"type": "input_file", "file_id": "file-report"},
            ],
        }
    ]
    assert create.await_args.kwargs["purpose"] == "user_data"
    assert create.await_args.kwargs["extra_query"] == {"provider": "lgos-files"}


async def test_openwebui_storage_upload_forwards_request_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = AsyncMock(
        return_value=SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": "stored-file"},
        )
    )

    class HttpClientContext:
        async def __aenter__(self) -> object:
            return SimpleNamespace(post=post)

        async def __aexit__(self, *_: object) -> None:
            pass

    monkeypatch.setattr(
        generic_files.httpx,
        "AsyncClient",
        lambda **_: HttpClientContext(),
    )
    request = SimpleNamespace(
        base_url="https://openwebui.example/",
        headers={"authorization": "Bearer browser-session"},
    )

    stored_id = await generic_files._store_openwebui_file(
        request,
        filename="chart.png",
        media_type="image/png",
        content=b"png-bytes",
        timeout=10,
    )

    assert stored_id == "stored-file"
    post.assert_awaited_once_with(
        "https://openwebui.example/api/v1/files/",
        params={"process": "false"},
        headers={"Authorization": "Bearer browser-session"},
        files={"file": ("chart.png", b"png-bytes", "image/png")},
    )


def test_interrupt_round_trip_uses_previous_response_id() -> None:
    call = interrupt_call()
    ask_user = _interrupts_to_ask_user(RESPONSE_ID, [call])
    answer = {
        "role": "tool",
        "tool_call_id": ask_user["id"],
        "content": json.dumps(
            {
                "status": "answered",
                "answers": {"resume_0": {"type": "option", "option_index": 0}},
            }
        ),
    }

    continuation = _ask_user_to_resume(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [ask_user],
            },
            answer,
        ]
    )

    assert continuation is not None
    outputs, previous_response_id = continuation
    assert previous_response_id == RESPONSE_ID
    assert outputs == [
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": "approve",
        }
    ]


async def test_interrupt_response_becomes_native_ask_user_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=response(interrupt_call()))
    install_client(monkeypatch, create=create)

    result = await generic_pipe.Pipe().pipe(body(stream=False))

    tool_call = result["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "ask_user"
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["output"][0]["type"] == "function_call"
    assert result["output"][0]["name"] == "ask_user"
    assert result["output"][0]["status"] == "pending"
    assert result["output"][0]["id"] == tool_call["id"]
    assert result["output"][0]["call_id"] == tool_call["id"]
    assert result["output"][0]["arguments"] == tool_call["function"]["arguments"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "malformation",
    [
        "missing-result",
        "duplicate-result",
        "mixed-calls",
        "missing-answer",
        "extra-answer",
    ],
)
async def test_invalid_interrupt_exchange_cannot_start_a_new_run(
    monkeypatch, stream, malformation
):
    ask_user = _interrupts_to_ask_user(RESPONSE_ID, [interrupt_call()])
    answers = {"resume_0": {"type": "option", "option_index": 0}}
    if malformation == "missing-answer":
        answers.clear()
    elif malformation == "extra-answer":
        answers["resume_1"] = {"type": "option", "option_index": 1}
    assistant = {"role": "assistant", "content": None, "tool_calls": [ask_user]}
    result = {
        "role": "tool",
        "tool_call_id": ask_user["id"],
        "content": json.dumps({"status": "answered", "answers": answers}),
    }
    messages = [assistant, result]
    if malformation == "missing-result":
        messages.pop()
    elif malformation == "duplicate-result":
        messages.append(result)
    elif malformation == "mixed-calls":
        assistant["tool_calls"].append(
            {"id": "call_other", "type": "function", "function": {"name": "other"}}
        )
    request = body(stream=stream)
    request["messages"].extend(messages)
    create = AsyncMock(return_value=final_response("Unexpected new run"))
    install_client(monkeypatch, create=create)

    output = await collect(generic_pipe.Pipe().pipe(request))

    assert "interrupt" in output[0]["error"]["detail"]
    create.assert_not_awaited()


async def test_interrupt_response_with_preliminary_text_preserves_content_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_part = final_response("Please confirm: ").output[0]
    create = AsyncMock(return_value=response(first_part, interrupt_call()))
    install_client(monkeypatch, create=create)

    result = await generic_pipe.Pipe().pipe(body(stream=False))

    assert result["choices"][0]["message"]["content"] == "Please confirm: "
    assert len(result["output"]) == 2
    assert result["output"][0]["type"] == "message"
    assert result["output"][0]["content"][0]["text"] == "Please confirm: "
    assert result["output"][1]["type"] == "function_call"
    assert result["output"][1]["name"] == "ask_user"
    assert result["output"][1]["status"] == "pending"


@pytest.mark.parametrize("phase", [None, "final_answer"])
async def test_non_streaming_answer_allows_optional_phase(monkeypatch, phase):
    completed = final_response("Answer")
    completed.output[0].phase = phase
    commentary = final_response("Working").output[0]
    commentary.phase = "commentary"
    completed.output.insert(0, commentary)
    install_client(monkeypatch, create=AsyncMock(return_value=completed))

    assert await generic_pipe.Pipe().pipe(body(stream=False)) == "Answer"


def test_transcript_preserves_assistant_phase_and_uses_native_file_parts():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "assistant", "content": "Working", "phase": "commentary"},
        {"role": "assistant", "content": "Answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this"},
                {"type": "input_file", "file_id": "file-123"},
            ],
        },
    ]

    assert _responses_input(messages) == [
        messages[0],
        messages[1],
        {"role": "assistant", "content": "Answer", "phase": "final_answer"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Read this"},
                {"type": "input_file", "file_id": "file-123"},
            ],
        },
    ]
