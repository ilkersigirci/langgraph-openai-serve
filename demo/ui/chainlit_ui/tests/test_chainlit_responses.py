"""Responses and durable display-file behavior for Chainlit."""

import importlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call

import pytest
from chainlit.context import init_http_context
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.parsed_response import ParsedResponseFunctionToolCall

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


@pytest.mark.parametrize("phase", [None, "final_answer"])
def test_final_answer_excludes_commentary(phase: str | None) -> None:
    commentary = ResponseOutputMessage(
        id="msg_commentary",
        content=[
            ResponseOutputText(
                annotations=[],
                logprobs=[],
                text="Rendering chart",
                type="output_text",
            )
        ],
        role="assistant",
        status="completed",
        type="message",
        phase="commentary",
    )
    final = ResponseOutputMessage(
        id="msg_final",
        content=[
            ResponseOutputText(
                annotations=[],
                logprobs=[],
                text="Chart ready.",
                type="output_text",
            )
        ],
        role="assistant",
        status="completed",
        type="message",
        phase=phase,
    )

    assert responses.final_answer(_response(commentary, final)) == "Chart ready."


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
    assert task_factory.call_args_list == [
        call(title="Generating audio", status=responses.cl.TaskStatus.RUNNING),
        call(title="Calculating embeddings", status=responses.cl.TaskStatus.RUNNING),
        call(title="Media ready", status=responses.cl.TaskStatus.RUNNING),
    ]
    assert [task.status for task in tasks] == [
        responses.cl.TaskStatus.DONE,
        responses.cl.TaskStatus.DONE,
        responses.cl.TaskStatus.DONE,
    ]
    assert task_list.add_task.await_args_list == [call(task) for task in tasks]
    assert task_list.status == "Done"
    assert task_list.send.await_count == 4


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
    assert task_list.send.await_count == 2


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
@pytest.mark.parametrize("provider", ["lgos-files", "litellm_proxy"])
async def test_display_plotly_persists_an_interactive_element(
    monkeypatch: pytest.MonkeyPatch,
    valid: bool,
    provider: str,
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
    monkeypatch.setattr(responses, "files_request", lambda: (client, provider))
    message = Mock(metadata=None, send=AsyncMock())
    message_factory = Mock(return_value=message)
    monkeypatch.setattr(responses.cl, "Message", message_factory)

    if not valid:
        with pytest.raises(ValueError):
            await responses.display_file(call)
        message_factory.assert_not_called()
        return

    output = await responses.display_file(call)

    content.assert_awaited_once_with("file-chart", extra_query={"provider": provider})
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
    last_text = first_text.model_copy(
        update={
            "id": "msg_final",
            "content": [
                ResponseOutputText(
                    type="output_text", text="Chart ready.", annotations=[]
                )
            ],
        }
    )
    first = _response(first_text, call)
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
    assistant = Mock(content="", send=AsyncMock(), update=AsyncMock())

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
        simple, "session_metadata", lambda: {"session_id": "thread-123"}
    )
    monkeypatch.setattr(simple, "model_request", lambda _: {"model": "plot"})
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(simple.openai_client.responses, "create", create)
    monkeypatch.setattr(simple, "_stream_response", stream)
    monkeypatch.setattr(simple, "display_file", AsyncMock(return_value=output))

    await simple._response_message(Mock(), "plot")

    assert assistant.content == "Here is the chart. Chart ready."
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
    monkeypatch.setattr(simple, "session_metadata", dict)
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
