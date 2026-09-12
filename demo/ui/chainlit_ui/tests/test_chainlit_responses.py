"""Responses and durable display-file behavior for Chainlit."""

import importlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from chainlit.context import init_http_context
from openai import AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.parsed_response import ParsedResponseFunctionToolCall
from openai.types.responses.response_output_text import AnnotationURLCitation

from lgos_chainlit.utils import responses


def _response(*output: object) -> Response:
    return Response.model_construct(status="completed", output=list(output))


def _display_call() -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="fc_chart",
        call_id="call_chart",
        name="display_file",
        arguments=(
            '{"file_id":"file-chart","filename":"chart.png",'
            '"media_type":"image/png","title":"Quarterly revenue",'
            '"alt":"Q4 is highest."}'
        ),
        status="completed",
        type="function_call",
    )


async def test_commentary_is_rendered_as_a_native_task_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_list = Mock(status="Ready", add_task=AsyncMock(), send=AsyncMock())
    task_list_factory = Mock(return_value=task_list)
    tasks = [Mock(), Mock(), Mock()]
    task_factory = Mock(side_effect=tasks)
    monkeypatch.setattr(responses.cl, "TaskList", task_list_factory)
    monkeypatch.setattr(responses.cl, "Task", task_factory)
    renderer = responses.CommentaryTaskList()

    await renderer.add("Generating audio")
    await renderer.add("Calculating embeddings")
    await renderer.add("Media ready")
    await renderer.complete()

    task_list_factory.assert_called_once_with()
    assert [item.kwargs["title"] for item in task_factory.call_args_list] == [
        "Generating audio",
        "Calculating embeddings",
        "Media ready",
    ]
    assert [task.status for task in tasks] == [
        responses.cl.TaskStatus.DONE,
        responses.cl.TaskStatus.DONE,
        responses.cl.TaskStatus.DONE,
    ]
    assert task_list.status == "Done"


@pytest.mark.parametrize("phase", [None, "final_answer"])
async def test_response_stream_routes_commentary_to_the_task_list(
    monkeypatch: pytest.MonkeyPatch,
    phase: str | None,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    completed = Response.model_construct(status="completed", output=[])
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(type="message", phase="commentary"),
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            delta="Generating ",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            delta="audio",
        ),
        SimpleNamespace(
            type="response.output_text.done",
            output_index=0,
            text="Generating audio",
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(type="message", phase=phase),
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=1,
            delta="Media ready.",
        ),
    ]
    stream = MagicMock()
    stream.__aiter__.return_value = iter(events)
    stream.get_final_response = AsyncMock(return_value=completed)
    stream_manager = MagicMock()
    stream_manager.__aenter__ = AsyncMock(return_value=stream)
    stream_manager.__aexit__ = AsyncMock(return_value=None)
    create_stream = Mock(return_value=stream_manager)
    monkeypatch.setattr(simple.openai_client.responses, "stream", create_stream)
    assistant_message = Mock(stream_token=AsyncMock())
    commentary_tasks = Mock(add=AsyncMock())

    response = await simple._stream_response(
        [],
        assistant_message,
        model="status-events",
        extra_headers=None,
        user="demo-user",
        metadata={},
        commentary_tasks=commentary_tasks,
    )

    assert response is completed
    commentary_tasks.add.assert_awaited_once_with("Generating audio")
    assistant_message.stream_token.assert_awaited_once_with("Media ready.")


async def test_stopped_commentary_marks_the_active_task_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_list = Mock(status="Ready", add_task=AsyncMock(), send=AsyncMock())
    task = Mock()
    monkeypatch.setattr(responses.cl, "TaskList", Mock(return_value=task_list))
    monkeypatch.setattr(responses.cl, "Task", Mock(return_value=task))
    renderer = responses.CommentaryTaskList()

    await renderer.add("Generating audio")
    await renderer.stop()

    assert task.status == responses.cl.TaskStatus.FAILED
    assert task_list.status == "Stopped"


@pytest.mark.parametrize("send_delta", [False, True])
async def test_streamed_refusal_is_visible_even_without_deltas(
    monkeypatch, send_delta, chainlit_context
):
    simple = importlib.import_module("lgos_chainlit.simple")
    refusal = ResponseOutputRefusal(
        type="refusal", refusal="Cannot answer this request."
    )
    message = ResponseOutputMessage(
        id="msg_refusal",
        type="message",
        role="assistant",
        status="completed",
        phase="final_answer",
        content=[refusal],
    )
    completed = _response(message)
    events = (
        [
            SimpleNamespace(
                type="response.refusal.delta", output_index=0, delta=refusal.refusal
            )
        ]
        if send_delta
        else []
    )
    stream = MagicMock()
    stream.__aiter__.return_value = iter(events)
    stream.get_final_response = AsyncMock(return_value=completed)
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=stream)
    monkeypatch.setattr(
        simple.openai_client.responses, "stream", Mock(return_value=manager)
    )
    assistant = Mock(stream_token=AsyncMock())

    await simple._stream_response(
        [],
        assistant,
        model="test",
        extra_headers=None,
        user="user",
        metadata={},
        commentary_tasks=Mock(),
    )
    assistant.stream_token.assert_awaited_once_with(refusal.refusal)
    assert responses.final_answer(completed) == refusal.refusal


def test_incomplete_response_reports_its_native_reason():
    from openai.types.responses.response import IncompleteDetails

    incomplete = _response().model_copy(
        update={
            "status": "incomplete",
            "incomplete_details": IncompleteDetails(reason="max_output_tokens"),
        }
    )
    with pytest.raises(RuntimeError, match="Response incomplete: max_output_tokens"):
        responses.raise_for_response(incomplete)


async def test_sdk_incomplete_event_reports_reason_without_waiting_for_completion(
    monkeypatch,
    chainlit_context,
):
    simple = importlib.import_module("lgos_chainlit.simple")
    incomplete = Response.model_construct(
        id="resp_partial",
        object="response",
        status="incomplete",
        output=[],
        incomplete_details={"reason": "max_output_tokens"},
    )
    initial = incomplete.model_copy(
        update={"status": "in_progress", "incomplete_details": None}
    )
    payloads = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": initial.model_dump(),
        },
        {
            "type": "response.incomplete",
            "sequence_number": 1,
            "response": incomplete.model_dump(),
        },
    ]
    wire = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in payloads
    )
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text=wire,
                )
            )
        ) as http_client,
        AsyncOpenAI(api_key="test", http_client=http_client) as client,
    ):
        monkeypatch.setattr(simple, "openai_client", client)
        with pytest.raises(
            RuntimeError, match="Response incomplete: max_output_tokens"
        ):
            await simple._stream_response(
                [],
                Mock(),
                model="test",
                extra_headers=None,
                user="user",
                metadata={},
                commentary_tasks=Mock(),
            )


@pytest.mark.parametrize("provider", ["lgos-files", "litellm_proxy"])
async def test_display_file_uses_a_persisted_native_image_message(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    download = SimpleNamespace(aread=AsyncMock(return_value=b"png-bytes"))
    content = AsyncMock(return_value=download)
    image = Mock()
    image_factory = Mock(return_value=image)
    message = Mock(metadata=None, send=AsyncMock())
    message_factory = Mock(return_value=message)
    client = SimpleNamespace(files=SimpleNamespace(content=content))
    monkeypatch.setattr(responses, "files_request", lambda: (client, provider))
    monkeypatch.setattr(responses.cl, "Image", image_factory)
    monkeypatch.setattr(responses.cl, "Message", message_factory)

    output = await responses.display_file(_display_call())

    content.assert_awaited_once_with("file-chart", extra_query={"provider": provider})
    image_factory.assert_called_once_with(
        name="chart.png",
        content=b"png-bytes",
        mime="image/png",
        display="inline",
    )
    message_factory.assert_called_once_with(
        content="Quarterly revenue",
        elements=[image],
    )
    assert message.metadata == {"lgos_chainlit.exclude_from_model_context": True}
    message.send.assert_awaited_once_with()
    assert output == {
        "type": "function_call_output",
        "call_id": "call_chart",
        "output": '{"displayed":true}',
    }


@pytest.mark.parametrize("valid", [True, False], ids=["plotly", "invalid-plotly"])
async def test_display_plotly_persists_an_interactive_element(
    monkeypatch: pytest.MonkeyPatch,
    valid: bool,
) -> None:
    init_http_context()
    call = _display_call()
    arguments = json.loads(call.arguments)
    arguments.update(
        filename="chart.plotly.json", media_type="application/vnd.plotly.v1+json"
    )
    call.arguments = json.dumps(arguments)
    chart = b'{"data":[{"type":"bar","x":["Q1","Q2"],"y":[120,180]}]}'
    download = SimpleNamespace(
        aread=AsyncMock(return_value=chart if valid else b"bad-json")
    )
    content = AsyncMock(return_value=download)
    client = SimpleNamespace(files=SimpleNamespace(content=content))
    monkeypatch.setattr(responses, "files_request", lambda: (client, "lgos-files"))
    message = Mock(metadata=None, send=AsyncMock())
    message_factory = Mock(return_value=message)
    monkeypatch.setattr(responses.cl, "Message", message_factory)

    if not valid:
        with pytest.raises(ValueError):
            await responses.display_file(call)
        message_factory.assert_not_called()
        return

    output = await responses.display_file(call)

    content.assert_awaited_once_with(
        "file-chart", extra_query={"provider": "lgos-files"}
    )
    element = message_factory.call_args.kwargs["elements"][0]
    assert isinstance(element, responses.cl.Plotly)
    assert element.display == "inline"
    assert json.loads(element.content)["data"] == json.loads(chart)["data"]
    assert message.metadata == {"lgos_chainlit.exclude_from_model_context": True}
    message.send.assert_awaited_once_with()
    assert output == {
        "type": "function_call_output",
        "call_id": "call_chart",
        "output": '{"displayed":true}',
    }


@pytest.mark.parametrize("parsed", [False, True])
def test_continuation_replays_only_wire_fields_before_its_small_output(
    parsed: bool,
) -> None:
    call = _display_call()
    expected_call = call.model_dump(mode="json", exclude_none=True)
    if parsed:
        call = ParsedResponseFunctionToolCall(
            **expected_call, parsed_arguments=json.loads(call.arguments)
        )
    response = _response(call)
    output = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }

    continuation = responses.continuation_input(response, [output])

    assert continuation == [expected_call, output]


@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_continuation_keeps_history_files_and_final_text(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    chainlit_context,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    call = _display_call()
    first_text = ResponseOutputMessage(
        id="msg_intro",
        role="assistant",
        type="message",
        status="completed",
        phase="final_answer",
        content=[
            ResponseOutputText(
                type="output_text", text="Here is the chart. ", annotations=[]
            )
        ],
    )
    last_answer = "Chart ready [source]"
    last_text = first_text.model_copy(
        update={
            "id": "msg_final",
            "content": [
                ResponseOutputText(
                    type="output_text",
                    text=last_answer,
                    annotations=[
                        AnnotationURLCitation(
                            type="url_citation",
                            url="https://example.com/chart",
                            title="Chart source",
                            start_index=last_answer.index("[source]"),
                            end_index=len(last_answer) - 1,
                        )
                    ],
                )
            ],
        }
    )
    commentary = first_text.model_copy(
        update={
            "id": "msg_commentary",
            "phase": "commentary",
            "content": [
                ResponseOutputText(
                    type="output_text", text="Rendering chart", annotations=[]
                )
            ],
        }
    )
    first = _response(commentary, first_text, call)
    pending = iter([first, _response(last_text)])
    requests = []
    history = [{"role": "system", "content": "Use the uploaded data."}]
    file_input = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Plot revenue"},
            {"type": "input_file", "file_id": "file-data"},
        ],
    }
    output = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }
    assistant = Mock(content="", elements=[], send=AsyncMock(), update=AsyncMock())

    async def create(**request):
        requests.append(deepcopy(request["input"]))
        return next(pending)

    async def stream(input_items, assistant_message, **_):
        completed = await create(input=input_items)
        assistant_message.content += responses.final_answer(completed)
        return completed

    monkeypatch.setattr(simple.cl, "Message", Mock(return_value=assistant))
    monkeypatch.setattr(simple, "text_only_chat_messages", lambda: history)
    monkeypatch.setattr(
        simple,
        "with_response_file_parts",
        AsyncMock(return_value=[*history, file_input]),
    )
    monkeypatch.setattr(simple, "streaming_enabled", lambda: streaming)
    monkeypatch.setattr(simple, "chat_settings_metadata", dict)
    monkeypatch.setattr(
        simple, "conversation_metadata", lambda: {"conversation_id": "thread-123"}
    )
    monkeypatch.setattr(simple, "model_request", lambda _: {"model": "plot"})
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(simple.openai_client.responses, "create", create)
    monkeypatch.setattr(simple, "_stream_response", stream)
    monkeypatch.setattr(simple, "display_file", AsyncMock(return_value=output))

    await simple._response_message(Mock(), "plot")

    assert assistant.content == "Here is the chart. Chart ready [source]"
    assert [
        (element.name, element.content, element.display)
        for element in assistant.elements
    ] == [("[source]", "[Open source](<https://example.com/chart>)", "side")]
    assert requests[0] == [*history, file_input]
    assert requests[1] == [
        *history,
        file_input,
        *(item.model_dump(mode="json", exclude_none=True) for item in first.output),
        output,
    ]


def test_transcript_labels_answers_and_preserves_explicit_phase():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Working", "phase": "commentary"},
        {"role": "assistant", "content": "Answer"},
    ]

    assert responses.response_input(messages) == [
        *messages[:3],
        {"role": "assistant", "content": "Answer", "phase": "final_answer"},
    ]


async def test_non_streaming_failure_does_not_display_files_or_send_success(
    monkeypatch,
    chainlit_context,
):
    simple = importlib.import_module("lgos_chainlit.simple")
    failed = _response(_display_call())
    failed.status = "failed"
    failed.error = SimpleNamespace(message="Graph failed")
    assistant = Mock(content="", send=AsyncMock())
    error = AsyncMock()
    display = AsyncMock()
    monkeypatch.setattr(simple.cl, "Message", Mock(return_value=assistant))
    monkeypatch.setattr(simple, "text_only_chat_messages", list)
    monkeypatch.setattr(simple, "with_response_file_parts", AsyncMock(return_value=[]))
    monkeypatch.setattr(simple, "streaming_enabled", lambda: False)
    monkeypatch.setattr(simple, "chat_settings_metadata", dict)
    monkeypatch.setattr(simple, "conversation_metadata", dict)
    monkeypatch.setattr(simple, "model_request", lambda _: {"model": "plot"})
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(
        simple.openai_client.responses, "create", AsyncMock(return_value=failed)
    )
    monkeypatch.setattr(simple, "display_file", display)
    monkeypatch.setattr(simple, "send_ui_message", error)

    await simple._response_message(Mock(), "plot")

    error.assert_awaited_once_with("Response failed: Graph failed")
    assistant.send.assert_not_awaited()
    display.assert_not_awaited()


@pytest.mark.parametrize(
    "model", ["hosted-tool", "lgos-a/hosted-tool", "lgos/lgos-a/hosted-tool"]
)
def test_hosted_tool_request_enables_server_execution(model: str) -> None:
    assert responses.response_tools(model) == [
        {"type": "custom", "name": "lgos_current_time"}
    ]
    assert responses.response_tools("lgos-a/simple-graph") == [
        responses.DISPLAY_FILE_TOOL
    ]


async def test_simple_ui_rejects_interrupt_calls_with_hitl_guidance(
    monkeypatch,
    chainlit_context,
) -> None:
    from lgos_chainlit.lgos_protocol import INTERRUPT_TOOL_NAME

    simple = importlib.import_module("lgos_chainlit.simple")
    interrupt_resp = _response(
        ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_interrupt_1",
            name=INTERRUPT_TOOL_NAME,
            arguments="{}",
            type="function_call",
        )
    )
    assistant = Mock(content="", send=AsyncMock())
    error = AsyncMock()
    display = AsyncMock()
    monkeypatch.setattr(simple.cl, "Message", Mock(return_value=assistant))
    monkeypatch.setattr(simple, "text_only_chat_messages", list)
    monkeypatch.setattr(simple, "with_response_file_parts", AsyncMock(return_value=[]))
    monkeypatch.setattr(simple, "streaming_enabled", lambda: False)
    monkeypatch.setattr(simple, "chat_settings_metadata", dict)
    monkeypatch.setattr(simple, "conversation_metadata", dict)
    monkeypatch.setattr(
        simple, "model_request", lambda _: {"model": "interruptible-approval"}
    )
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(
        simple.openai_client.responses, "create", AsyncMock(return_value=interrupt_resp)
    )
    monkeypatch.setattr(simple, "display_file", display)
    monkeypatch.setattr(simple, "send_ui_message", error)

    await simple._response_message(Mock(), "interruptible-approval")

    assert error.await_count == 1
    assert "DEMO_CHAINLIT_UI_FILE=hitl" in error.await_args[0][0]
    display.assert_not_awaited()
