"""Responses and durable display-file behavior for Chainlit."""

import importlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call

import httpx2
import pytest
from chainlit.context import init_http_context
from openai import AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import AnnotationURLCitation

from lgos_chainlit import display_files


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
async def test_response_stream_routes_commentary_to_the_task_list(
    monkeypatch: pytest.MonkeyPatch,
    phase: str | None,
    chainlit_context,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
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
    monkeypatch.setattr(chat.openai_client.responses, "stream", create_stream)
    assistant_message = Mock(stream_token=AsyncMock())
    commentary_tasks = Mock(add=AsyncMock())

    response = await chat._stream_response(
        [],
        assistant_message,
        model="status-events",
        extra_headers=None,
        user="demo-user",
        metadata={},
        commentary_tasks=commentary_tasks,
    )

    assert response is completed
    assert commentary_tasks.add.await_args_list == [call("Generating audio")]
    assistant_message.stream_token.assert_awaited_once_with("Media ready.")


@pytest.mark.parametrize("send_delta", [False, True])
async def test_streamed_refusal_is_visible_even_without_deltas(
    monkeypatch, send_delta, chainlit_context
):
    chat = importlib.import_module("lgos_chainlit.chat")
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
        chat.openai_client.responses, "stream", Mock(return_value=manager)
    )
    assistant = Mock(stream_token=AsyncMock())

    await chat._stream_response(
        [],
        assistant,
        model="test",
        extra_headers=None,
        user="user",
        metadata={},
        commentary_tasks=Mock(),
    )
    assistant.stream_token.assert_awaited_once_with(refusal.refusal)


async def test_sdk_incomplete_event_reports_reason_without_waiting_for_completion(
    monkeypatch,
    chainlit_context,
):
    chat = importlib.import_module("lgos_chainlit.chat")
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
        httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text=wire,
                )
            )
        ) as http_client,
        AsyncOpenAI(api_key="test", http_client=http_client) as client,
    ):
        monkeypatch.setattr(chat, "openai_client", client)
        with pytest.raises(
            RuntimeError, match="Response incomplete: max_output_tokens"
        ):
            await chat._stream_response(
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
    monkeypatch.setattr(display_files, "files_request", lambda: (client, provider))
    monkeypatch.setattr(display_files.cl, "Image", image_factory)
    monkeypatch.setattr(display_files.cl, "Message", message_factory)

    output = await display_files.display_file(_display_call())

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
    monkeypatch.setattr(display_files, "files_request", lambda: (client, "lgos-files"))
    message = Mock(metadata=None, send=AsyncMock())
    message_factory = Mock(return_value=message)
    monkeypatch.setattr(display_files.cl, "Message", message_factory)

    if not valid:
        with pytest.raises(ValueError):
            await display_files.display_file(call)
        message_factory.assert_not_called()
        return

    output = await display_files.display_file(call)

    content.assert_awaited_once_with(
        "file-chart", extra_query={"provider": "lgos-files"}
    )
    element = message_factory.call_args.kwargs["elements"][0]
    assert isinstance(element, display_files.cl.Plotly)
    assert element.display == "inline"
    assert json.loads(element.content)["data"] == json.loads(chart)["data"]
    assert message.metadata == {"lgos_chainlit.exclude_from_model_context": True}
    message.send.assert_awaited_once_with()
    assert output == {
        "type": "function_call_output",
        "call_id": "call_chart",
        "output": '{"displayed":true}',
    }


@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_continuation_keeps_history_files_and_final_text(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    chainlit_context,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
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
    server_call = ResponseCustomToolCall.model_validate(
        {
            "type": "custom_tool_call",
            "id": "ctc_package",
            "call_id": "call_package",
            "name": "lgos_package_version",
            "input": "openai",
            "status": "completed",
        }
    )
    server_output = ResponseCustomToolCallOutputItem(
        type="custom_tool_call_output",
        id="ctco_package",
        call_id="call_package",
        output="openai==installed-version",
        status="completed",
    )
    first = _response(commentary, first_text, server_call, server_output, call)
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
        assistant_message.content += chat.final_answer(completed)
        return completed

    monkeypatch.setattr(chat.cl, "Message", Mock(return_value=assistant))
    monkeypatch.setattr(chat, "text_only_chat_messages", lambda: history)
    monkeypatch.setattr(
        chat,
        "with_response_file_parts",
        AsyncMock(return_value=[*history, file_input]),
    )
    monkeypatch.setattr(chat, "streaming_enabled", lambda: streaming)
    monkeypatch.setattr(chat, "chat_settings_metadata", dict)
    monkeypatch.setattr(
        chat, "conversation_metadata", lambda: {"conversation_id": "thread-123"}
    )
    monkeypatch.setattr(chat, "model_request", lambda _: {"model": "plot"})
    monkeypatch.setattr(chat, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(chat.openai_client.responses, "create", create)
    monkeypatch.setattr(chat, "_stream_response", stream)
    display = AsyncMock(return_value=output)
    monkeypatch.setattr(chat, "display_file", display)

    await chat._response_message(Mock(), "plot")

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
    display.assert_awaited_once_with(call)


async def test_non_streaming_failure_does_not_display_files_or_send_success(
    monkeypatch,
    chainlit_context,
):
    chat = importlib.import_module("lgos_chainlit.chat")
    failed = _response(_display_call())
    failed.status = "failed"
    failed.error = SimpleNamespace(message="Graph failed")
    assistant = Mock(content="", send=AsyncMock())
    error = AsyncMock()
    display = AsyncMock()
    monkeypatch.setattr(chat.cl, "Message", Mock(return_value=assistant))
    monkeypatch.setattr(chat, "text_only_chat_messages", list)
    monkeypatch.setattr(chat, "with_response_file_parts", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat, "streaming_enabled", lambda: False)
    monkeypatch.setattr(chat, "chat_settings_metadata", dict)
    monkeypatch.setattr(chat, "conversation_metadata", dict)
    monkeypatch.setattr(chat, "model_request", lambda _: {"model": "plot"})
    monkeypatch.setattr(chat, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(
        chat.openai_client.responses, "create", AsyncMock(return_value=failed)
    )
    monkeypatch.setattr(chat, "display_file", display)
    monkeypatch.setattr(chat, "send_ui_message", error)

    await chat._response_message(Mock(), "plot")

    error.assert_awaited_once_with("Response failed: Graph failed")
    assistant.send.assert_not_awaited()
    display.assert_not_awaited()


async def test_interrupt_calls_are_delegated_to_the_durable_workflow(
    monkeypatch,
    chainlit_context,
) -> None:
    from lgos_chainlit.lgos_protocol import INTERRUPT_TOOL_NAME

    chat = importlib.import_module("lgos_chainlit.chat")
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
    workflow = SimpleNamespace(publish=AsyncMock())
    display = AsyncMock()
    monkeypatch.setattr(chat.cl, "Message", Mock(return_value=assistant))
    monkeypatch.setattr(chat, "text_only_chat_messages", list)
    monkeypatch.setattr(chat, "with_response_file_parts", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat, "streaming_enabled", lambda: False)
    monkeypatch.setattr(chat, "chat_settings_metadata", dict)
    monkeypatch.setattr(chat, "conversation_metadata", dict)
    monkeypatch.setattr(
        chat, "model_request", lambda _: {"model": "interruptible-approval"}
    )
    monkeypatch.setattr(chat, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(
        chat.openai_client.responses, "create", AsyncMock(return_value=interrupt_resp)
    )
    monkeypatch.setattr(chat, "interrupt_workflow", workflow)
    monkeypatch.setattr(chat, "display_file", display)

    await chat._response_message(Mock(), "interruptible-approval")

    workflow.publish.assert_awaited_once_with(
        interrupt_resp,
        model_id="interruptible-approval",
    )
    display.assert_not_awaited()


async def test_pending_interrupt_blocks_a_new_model_turn(monkeypatch) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    workflow = SimpleNamespace(block_new_message=AsyncMock(return_value=True))
    respond = AsyncMock()
    monkeypatch.setattr(chat, "interrupt_workflow", workflow)
    monkeypatch.setattr(chat, "_response_message", respond)
    message = Mock()

    await chat.on_message(message)

    workflow.block_new_message.assert_awaited_once_with(message)
    respond.assert_not_awaited()


async def test_interrupt_action_submits_only_the_ui_reference(
    monkeypatch,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    workflow = SimpleNamespace(submit=AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "interrupt_workflow", workflow)
    action = chat.cl.Action(
        name=chat.INTERRUPT_ACTION_NAME,
        payload={
            "step_id": "step-review",
            "element_id": "element-review",
            "revision": "resp-review",
            "outputs": ["approve"],
        },
    )

    result = await chat.on_interrupt_submit(action)

    assert result == {"ok": True}
    workflow.submit.assert_awaited_once_with(
        step_id="step-review",
        element_id="element-review",
        revision="resp-review",
        outputs=["approve"],
    )


async def test_interrupt_action_rejects_an_invalid_browser_payload(
    monkeypatch,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    workflow = SimpleNamespace(submit=AsyncMock())
    monkeypatch.setattr(chat, "interrupt_workflow", workflow)
    action = chat.cl.Action(
        name=chat.INTERRUPT_ACTION_NAME,
        payload={
            "step_id": "step-review",
            "element_id": "element-review",
            "revision": "resp-review",
            "outputs": "approve",
        },
    )

    result = await chat.on_interrupt_submit(action)

    assert result == {"ok": False, "error": "Invalid human-review submission."}
    workflow.submit.assert_not_awaited()


async def test_interrupt_action_returns_a_stale_submission_error(
    monkeypatch,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    workflow = SimpleNamespace(
        submit=AsyncMock(
            side_effect=chat.InvalidHitlSubmissionError(
                "This human-review revision is stale."
            )
        )
    )
    monkeypatch.setattr(chat, "interrupt_workflow", workflow)
    action = chat.cl.Action(
        name=chat.INTERRUPT_ACTION_NAME,
        payload={
            "step_id": "step-review",
            "element_id": "element-review",
            "revision": "resp-review",
            "outputs": ["approve"],
        },
    )

    result = await chat.on_interrupt_submit(action)

    assert result == {
        "ok": False,
        "error": "This human-review revision is stale.",
    }


async def test_interrupt_continuation_keeps_response_cursor_and_request_context(
    monkeypatch,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    completed = _response()
    create = AsyncMock(return_value=completed)
    monkeypatch.setattr(chat.openai_client.responses, "create", create)
    monkeypatch.setattr(
        chat,
        "model_request",
        lambda _: {
            "model": "interruptible-approval",
            "extra_headers": {"x-model-provider": "lgos-a"},
        },
    )
    monkeypatch.setattr(
        chat,
        "chat_settings_metadata",
        lambda: {"lgos_settings": '{"audience":"expert"}'},
    )
    monkeypatch.setattr(
        chat,
        "conversation_metadata",
        lambda: {"conversation_id": "thread-123"},
    )
    monkeypatch.setattr(chat, "response_tools", lambda: [{"type": "web_search"}])
    monkeypatch.setattr(chat, "authenticated_user_identifier", lambda: "demo-user")
    input_items = [
        {
            "type": "function_call_output",
            "call_id": "call-review",
            "output": "approve",
        }
    ]

    response = await chat._continue_interrupt_response(
        input_items,
        model_id="lgos-a/interruptible-approval",
        previous_response_id="resp-review",
    )

    assert response is completed
    assert create.await_args.kwargs == {
        "model": "interruptible-approval",
        "extra_headers": {"x-model-provider": "lgos-a"},
        "input": input_items,
        "previous_response_id": "resp-review",
        "store": False,
        "tools": [{"type": "web_search"}],
        "user": "demo-user",
        "metadata": {
            "lgos_settings": '{"audience":"expert"}',
            "conversation_id": "thread-123",
        },
    }
