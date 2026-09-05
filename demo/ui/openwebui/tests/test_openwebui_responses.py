"""Responses-only Open WebUI Function behavior."""

import json
import sys
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
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

MODEL_ID = "interruptible-approval"
QUALIFIED_MODEL_ID = f"generic.{MODEL_ID}"


def response(*output: object) -> Response:
    return Response.model_construct(status="completed", output=list(output))


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
        "langgraph_interrupt",
        {
            "run_id": "725c277a-f6d5-4c52-95eb-8c09e91f7a7c",
            "state_token": "state-token-1",
            "payload": {
                "question": "Approve refund?",
                "choices": ["approve", "reject"],
                "allow_other": False,
            },
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


async def test_pipe_lists_both_litellm_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_urls = []

    @asynccontextmanager
    async def catalog_client(*, base_url: str, **_: object) -> AsyncIterator[object]:
        catalog_urls.append(base_url)
        model_prefix = base_url.rsplit("/", maxsplit=1)[-1]
        yield SimpleNamespace(
            models=SimpleNamespace(
                list=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[
                            SimpleNamespace(
                                id="simple-graph",
                                owned_by="langgraph-openai-serve",
                            )
                        ]
                    )
                )
            ),
            model_prefix=model_prefix,
        )

    monkeypatch.setattr(generic_pipe, "_client", catalog_client)

    models = await generic_pipe.Pipe().pipes()

    assert catalog_urls == [
        "http://lgos-litellm:4000/v1/lgos-a",
        "http://lgos-litellm:4000/v1/lgos-b",
    ]
    assert models == [
        {"id": "lgos-a/simple-graph", "name": "Generic / lgos-a/simple-graph"},
        {"id": "lgos-b/simple-graph", "name": "Generic / lgos-b/simple-graph"},
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_dir = Path(generic_pipe.__file__).parent
    source = bundle_function(function_dir)
    module = ModuleType("bundled_generic")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "<generic>", "exec"), module.__dict__)
    create = AsyncMock(return_value=final_response("Bundle answer."))
    module._client = lambda **_: FakeClient(create=create)

    result = await module.Pipe().pipe(body(stream=False))

    assert result == "Bundle answer."
    request = create.await_args.kwargs
    assert request["input"] == [{"role": "user", "content": "Refund ORDER-123"}]
    assert request["store"] is False


@pytest.mark.parametrize("deltas", [False, True])
async def test_bundled_stream_keeps_sse_looking_text_as_content(
    monkeypatch: pytest.MonkeyPatch, deltas: bool
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

    source = bundle_function(Path(generic_pipe.__file__).parent)
    module = ModuleType("bundled_generic")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "<generic>", "exec"), module.__dict__)
    module._client = lambda **_: FakeClient(stream=scripted_stream)

    output = await collect(module.Pipe().pipe(body(stream=True)))

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
        __metadata__={"chat_id": "thread-123"},
        __user__={"id": "user-123"},
    )

    assert result == "Approved."
    request = create.await_args.kwargs
    assert request["model"] == "lgos-a/interruptible-approval"
    assert "extra_headers" not in request
    assert request["input"] == [{"role": "user", "content": "Refund ORDER-123"}]
    assert request["store"] is False
    assert request["user"] == "user-123"
    assert request["metadata"] == {"session_id": "thread-123"}
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
    assert request["metadata"]["session_id"] == "thread-123"
    assert json.loads(request["metadata"]["langgraph_runtime_settings"]) == {
        "use_history": settings.use_history,
        "audience": settings.audience,
    }
    assert request["input"] == [{"role": "user", "content": "Hello"}]


async def test_bifrost_request_uses_native_responses_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_urls = []

    @asynccontextmanager
    async def client(*, base_url: str, **_: object) -> AsyncIterator[object]:
        base_urls.append(base_url)
        yield FakeClient(create=create)

    create = AsyncMock(return_value=final_response("Approved."))
    monkeypatch.setattr(generic_pipe, "_client", client)
    pipe = generic_pipe.Pipe()
    pipe.valves.OPENAI_GATEWAY_TYPE = "bifrost"
    pipe.valves.OPENAI_GATEWAY_BASE_URL = "https://bifrost.example"

    result = await pipe.pipe(
        {
            **body(stream=False),
            "model": "generic.lgos-a/interruptible-approval",
        },
        __metadata__={"chat_id": "thread-123"},
        __user__={"id": "user-123"},
    )

    assert result == "Approved."
    assert base_urls == ["https://bifrost.example/openai/v1"]
    request = create.await_args.kwargs
    assert request["model"] == "interruptible-approval"
    assert request["extra_headers"] == {"x-model-provider": "lgos-a"}


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
        [commentary, commentary_done, final_added, final_delta],
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
    emit.assert_awaited_once_with(
        {
            "type": "status",
            "data": {"description": "Checking policy", "done": True},
        }
    )


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
        end_index=text.index(cited_text) + len(cited_text) - 1,
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
async def test_display_file_continuation_preserves_input_and_all_final_text(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
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
    first = response(*final_response("Here is the chart. ").output, call)
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
    assert requests[0]["input"] == request_body["messages"]
    assert requests[1]["input"] == [
        *request_body["messages"],
        *expected_output,
        output,
    ]
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


@pytest.mark.parametrize("provider", ["lgos-files", "litellm_proxy"])
@pytest.mark.parametrize(
    "invalid_content",
    [None, b"bad-json", b'{"data":{}}', b'{"data":[],"layout":[]}'],
    ids=["plotly", "invalid-json", "invalid-data", "invalid-layout"],
)
async def test_display_plotly_emits_a_persistent_interactive_embed(
    monkeypatch: pytest.MonkeyPatch,
    invalid_content: bytes | None,
    provider: str,
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
        "provider": provider,
    }
    if invalid_content is not None:
        with pytest.raises(ValueError):
            await generic_files._handle_display_file(call, emit, object(), **kwargs)
        emit.assert_not_awaited()
        return

    output = await generic_files._handle_display_file(call, emit, object(), **kwargs)

    files_client.files.content.assert_awaited_once_with(
        "file-chart", extra_query={"provider": provider}
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


def test_interrupt_round_trip_uses_responses_function_items() -> None:
    call = interrupt_call()
    ask_user = _interrupts_to_ask_user([call])
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

    replay = _ask_user_to_resume(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [ask_user],
            },
            answer,
        ]
    )

    assert replay is not None
    assert replay[0]["type"] == "function_call"
    assert replay[0]["call_id"] == call.call_id
    assert replay[1] == {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"resume":"approve"}',
    }


async def test_interrupt_response_becomes_native_ask_user_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = AsyncMock(return_value=response(interrupt_call()))
    install_client(monkeypatch, create=create)

    result = await generic_pipe.Pipe().pipe(body(stream=False))

    tool_call = result["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "ask_user"
    assert result["choices"][0]["finish_reason"] == "tool_calls"


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
