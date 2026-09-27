"""Responses turns, client tools, background runs, and interrupt reviews."""

import asyncio
import json
from uuid import UUID

import anyio
import chainlit as cl
import httpx2
import pytest
from chainlit_utils.chat.hitl import HITL_CONTROL_PROP
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
)
from openai.types.responses.response_output_text import AnnotationURLCitation

from lgos_chainlit import chat
from lgos_chainlit.chat_settings import BACKGROUND_SETTING_ID, STREAMING_SETTING_ID
from lgos_chainlit.display_files import DISPLAY_FILE_TOOL
from lgos_chainlit.gateway import gateway_config
from lgos_chainlit.interrupts import INTERRUPT_ACTION_NAME
from tests.support import (
    function_call,
    message,
    reply,
    response,
    select_profile,
    streamed,
    transcript,
    user_message,
)

EXCLUDED_KEY = "lgos_chainlit.exclude_from_model_context"
DISPLAY_CHART = function_call(
    "display_file",
    json.dumps(
        {
            "file_id": "file-chart",
            "filename": "chart.png",
            "media_type": "image/png",
            "title": "Quarterly revenue",
            "alt": "Q4 is highest.",
        }
    ),
    call_id="call_chart",
)


@pytest.mark.parametrize("phase", [None, "final_answer"])
async def test_streamed_commentary_goes_to_the_task_list(
    chainlit_context,
    fake_gateway,
    task_lists,
    phase: str | None,
) -> None:
    chainlit_context.session.chat_profile = "lgos-a/status-events"
    fake_gateway.replies.append(
        streamed(
            response(
                message("Generating audio", id="msg_status", phase="commentary"),
                message("Media ready.", phase=phase),
            )
        )
    )

    await chat.on_message(user_message("Make audio."))

    assert fake_gateway.bodies("/v1/responses")[0]["stream"] is True
    assert transcript() == ["Make audio.", "Media ready."]
    assert task_lists[-1] == {
        "status": "Done",
        "tasks": [{"title": "Generating audio", "status": "done", "forId": None}],
    }


@pytest.mark.parametrize("deltas", [True, False], ids=["deltas", "final-only"])
async def test_streamed_refusal_is_visible(
    chainlit_context,
    fake_gateway,
    deltas: bool,
) -> None:
    chainlit_context.session.chat_profile = "lgos-a/simple-graph"
    refusal = ResponseOutputMessage(
        id="msg_refusal",
        type="message",
        role="assistant",
        status="completed",
        phase="final_answer",
        content=[ResponseOutputRefusal(type="refusal", refusal="I cannot help.")],
    )
    fake_gateway.replies.append(streamed(response(refusal), deltas=deltas))

    await chat.on_message(user_message("Help me."))

    assert transcript() == ["Help me.", "I cannot help."]


async def test_incomplete_stream_reports_its_reason(
    chainlit_context,
    fake_gateway,
) -> None:
    chainlit_context.session.chat_profile = "lgos-a/simple-graph"
    fake_gateway.replies.append(
        streamed(
            response(
                message("Partial"),
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            )
        )
    )

    await chat.on_message(user_message("Write a long essay."))

    assert transcript()[-2:] == [
        "Partial",
        "Response failed: Response incomplete: max_output_tokens.",
    ]
    assert cl.chat_context.get()[-2].metadata == {EXCLUDED_KEY: True}


@pytest.mark.parametrize("streaming", [False, True], ids=["create", "stream"])
async def test_client_tool_continuation_replays_the_turn_and_keeps_answer_text(
    chainlit_context,
    fake_gateway,
    streaming: bool,
) -> None:
    session = chainlit_context.session
    session.chat_profile = "lgos-a/persistent-plot-agent"
    session.chat_settings[STREAMING_SETTING_ID] = streaming
    first = response(
        message("Rendering chart", id="msg_status", phase="commentary"),
        message("Here is the chart. ", id="msg_intro"),
        ResponseCustomToolCall(
            type="custom_tool_call",
            id="ctc_package",
            call_id="call_package",
            name="lgos_package_version",
            input="openai",
            status="completed",
        ),
        ResponseCustomToolCallOutputItem(
            type="custom_tool_call_output",
            id="ctco_package",
            call_id="call_package",
            output="openai==installed-version",
            status="completed",
        ),
        DISPLAY_CHART,
    )
    answer = "Chart ready [source]"
    citation = AnnotationURLCitation(
        type="url_citation",
        url="https://example.com/chart",
        title="Chart source",
        start_index=answer.index("[source]"),
        end_index=len(answer) - 1,
    )
    deliver = streamed if streaming else reply
    fake_gateway.replies += [
        deliver(first),
        httpx2.Response(200, content=b"png-bytes"),
        deliver(response(message(answer, id="msg_final", annotations=[citation]))),
    ]
    user_message("What was revenue?")
    cl.chat_context.add(cl.Message(content="Revenue grew."))

    await chat.on_message(user_message("Plot revenue."))

    history = [
        {"role": "user", "content": "What was revenue?"},
        {"role": "assistant", "content": "Revenue grew.", "phase": "final_answer"},
        {"role": "user", "content": "Plot revenue."},
    ]
    initial, continuation = fake_gateway.bodies("/v1/responses")
    assert initial == {
        "model": "lgos-a/persistent-plot-agent",
        "input": history,
        "tools": [DISPLAY_FILE_TOOL],
        "user": "demo-user",
        "metadata": {"conversation_id": session.thread_id},
        "store": False,
        **({"stream": True} if streaming else {}),
    }
    assert continuation["input"] == [
        *history,
        *(item.model_dump(mode="json", exclude_none=True) for item in first.output),
        {
            "type": "function_call_output",
            "call_id": "call_chart",
            "output": '{"displayed":true}',
        },
    ]
    download = fake_gateway.requests[1]
    assert (download.url.path, dict(download.url.params)) == (
        "/v1/files/file-chart/content",
        {"provider": "litellm_proxy"},
    )
    assert {request.headers["User-Agent"] for request in fake_gateway.requests} == {
        "lgos-chainlit"
    }

    *_, chart, final = cl.chat_context.get()
    assert (chart.content, chart.metadata) == (
        "Quarterly revenue",
        {EXCLUDED_KEY: True},
    )
    assert [(image.type, image.mime) for image in chart.elements] == [
        ("image", "image/png")
    ]
    assert final.content == "Here is the chart. Chart ready [source]"
    assert [(link.name, link.content) for link in final.elements] == [
        ("[source]", "[Open source](<https://example.com/chart>)")
    ]


async def test_failed_response_reports_the_error_without_running_tools(
    chainlit_context,
    fake_gateway,
) -> None:
    chainlit_context.session.chat_profile = "lgos-a/persistent-plot-agent"
    chainlit_context.session.chat_settings[STREAMING_SETTING_ID] = False
    fake_gateway.replies.append(
        reply(
            response(
                DISPLAY_CHART,
                status="failed",
                error={"code": "server_error", "message": "Graph failed"},
            )
        )
    )

    await chat.on_message(user_message("Plot revenue."))

    assert transcript() == ["Plot revenue.", "Response failed: Graph failed"]
    assert len(fake_gateway.requests) == 1


async def _select_background_profile(fake_gateway, model_id: str) -> None:
    await select_profile(fake_gateway, model_id, features=["background"])
    cl.user_session.get("chat_settings")[BACKGROUND_SETTING_ID] = True


def _use_gateway(monkeypatch: pytest.MonkeyPatch, gateway_type: str) -> str:
    """Send Responses through ``gateway_type``'s route and return its path."""
    config = gateway_config(gateway_type, "https://gateway.example")
    monkeypatch.setattr(chat, "gateway", config)
    monkeypatch.setattr(
        chat,
        "responses_client",
        chat.responses_client.with_options(base_url=config.responses_base_url),
    )
    return httpx2.URL(config.responses_base_url).path + "/responses"


BACKGROUND_GATEWAYS = pytest.mark.parametrize(
    ("gateway_type", "lifecycle_query"),
    [("litellm", {}), ("bifrost", {"provider": "lgos-b"})],
)


@BACKGROUND_GATEWAYS
async def test_background_response_is_polled_until_complete(
    chainlit_context,
    fake_gateway,
    task_lists,
    monkeypatch: pytest.MonkeyPatch,
    gateway_type: str,
    lifecycle_query: dict[str, str],
) -> None:
    responses_path = _use_gateway(monkeypatch, gateway_type)
    monkeypatch.setattr(chat, "BACKGROUND_POLL_SECONDS", 0)
    await _select_background_profile(fake_gateway, "lgos-b/background-report")
    fake_gateway.replies += [
        reply(response(id="resp_bg", status="queued")),
        reply(response(id="resp_bg", status="in_progress")),
        reply(response(message("Report ready."), id="resp_bg")),
    ]

    await chat.on_message(user_message("Build the report."))

    [create] = fake_gateway.bodies(responses_path)
    assert (create["background"], create["store"]) == (True, True)
    assert create["metadata"] == {"conversation_id": chainlit_context.session.thread_id}
    create_request = fake_gateway.requests[1]
    idempotency_key = (
        create_request.headers["Idempotency-Key"]
        if gateway_type == "bifrost"
        else create["extra_headers"]["Idempotency-Key"]
    )
    UUID(idempotency_key)
    assert [
        (request.method, request.url.path, dict(request.url.params))
        for request in fake_gateway.requests[2:]
    ] == [("GET", f"{responses_path}/resp_bg", lifecycle_query)] * 2
    assert task_lists[-1] == {
        "status": "Done",
        "tasks": [
            {"title": "Background response queued", "status": "done", "forId": None},
            {
                "title": "Background response in progress",
                "status": "done",
                "forId": None,
            },
        ],
    }
    assert transcript()[-1] == "Report ready."


@BACKGROUND_GATEWAYS
async def test_stopped_turn_cancels_its_background_response(
    chainlit_context,
    fake_gateway,
    monkeypatch: pytest.MonkeyPatch,
    gateway_type: str,
    lifecycle_query: dict[str, str],
) -> None:
    responses_path = _use_gateway(monkeypatch, gateway_type)
    await _select_background_profile(fake_gateway, "lgos-b/background-report")
    polling = asyncio.Event()

    def in_progress(_request: httpx2.Request) -> httpx2.Response:
        polling.set()
        return reply(response(id="resp_bg", status="in_progress"))

    fake_gateway.replies += [
        reply(response(id="resp_bg", status="queued")),
        in_progress,
        reply(response(id="resp_bg", status="cancelled")),
    ]

    turn = asyncio.create_task(chat.on_message(user_message("Build the report.")))
    with anyio.fail_after(5):
        await polling.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    cancel = fake_gateway.requests[-1]
    assert (cancel.method, cancel.url.path, dict(cancel.url.params)) == (
        "POST",
        f"{responses_path}/resp_bg/cancel",
        lifecycle_query,
    )


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
async def test_interrupt_review_resumes_with_the_turn_request_context(
    chainlit_context,
    fake_gateway,
    monkeypatch: pytest.MonkeyPatch,
    background: bool,
) -> None:
    monkeypatch.setattr(chat, "BACKGROUND_POLL_SECONDS", 0)
    model_id = "lgos-a/interruptible-approval"
    await select_profile(fake_gateway, model_id, features=["background"])
    cl.user_session.get("chat_settings")[BACKGROUND_SETTING_ID] = background
    review_call = function_call(
        "lgos_interrupt",
        json.dumps(
            {
                "question": "Approve refund?",
                "request": "ORDER-123",
                "choices": ["approve", "reject"],
            }
        ),
        call_id="call_lg_review",
    )
    interrupted = response(review_call, id="resp_lg_review")
    approved = response(message("Refund approved."), id="resp_lg_done")
    fake_gateway.replies += (
        [reply(response(id=interrupted.id, status="queued")), reply(interrupted)]
        if background
        else [streamed(interrupted)]
    )

    await chat.on_message(user_message("Refund order ORDER-123."))

    review = cl.chat_context.get()[-1]
    assert review.content == "Approve refund?\n\nRequest: ORDER-123"
    control = review.elements[0].props[HITL_CONTROL_PROP]

    requests_before_block = len(fake_gateway.requests)
    await chat.on_message(user_message("Something else."))

    assert len(fake_gateway.requests) == requests_before_block
    assert transcript()[-1] == (
        "Resolve the pending interrupt before starting another request."
    )

    fake_gateway.replies += (
        [reply(response(id=approved.id, status="queued")), reply(approved)]
        if background
        else [streamed(approved)]
    )
    result = await chat.on_interrupt_submit(
        cl.Action(
            name=INTERRUPT_ACTION_NAME,
            payload={
                "step_id": control["step_id"],
                "element_id": control["element_id"],
                "revision": control["revision"],
                "outputs": ["approve"],
            },
        )
    )

    assert result == {"ok": True}
    resume = fake_gateway.bodies("/v1/responses")[-1]
    assert {
        key: resume[key]
        for key in ("model", "previous_response_id", "input", "tools", "user")
    } == {
        "model": model_id,
        "previous_response_id": "resp_lg_review",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_lg_review",
                "output": "approve",
            }
        ],
        "tools": [],
        "user": "demo-user",
    }
    assert resume["metadata"] == {"conversation_id": chainlit_context.session.thread_id}
    assert (resume.get("background", False), resume["store"]) == (
        background,
        background,
    )
    assert resume.get("stream", False) is not background
    assert review.elements == []
    assert transcript()[-1] == "Refund approved."


@pytest.mark.parametrize("streaming", [False, True], ids=["create", "stream"])
async def test_text_before_a_pause_stays_in_the_conversation(
    chainlit_context,
    fake_gateway,
    streaming: bool,
) -> None:
    await select_profile(fake_gateway, "lgos-a/interruptible-approval")
    chainlit_context.session.chat_settings[STREAMING_SETTING_ID] = streaming
    paused = response(
        message("I checked ORDER-123."),
        function_call(
            "lgos_interrupt",
            json.dumps({"question": "Approve refund?", "choices": ["approve"]}),
            call_id="call_lg_review",
        ),
        id="resp_lg_review",
    )
    fake_gateway.replies.append(streamed(paused) if streaming else reply(paused))

    await chat.on_message(user_message("Refund order ORDER-123."))

    assert transcript() == [
        "Refund order ORDER-123.",
        "I checked ORDER-123.",
        "Approve refund?",
    ]


async def test_client_tool_after_review_finishes_the_resumed_turn(
    chainlit_context,
    fake_gateway,
) -> None:
    chainlit_context.session.chat_profile = "lgos-a/persistent-plot-agent"
    review_call = function_call(
        "lgos_interrupt",
        json.dumps({"question": "Save the chart?", "choices": ["approve"]}),
        call_id="call_lg_review",
    )
    fake_gateway.replies.append(streamed(response(review_call, id="resp_lg_review")))
    await chat.on_message(user_message("Plot revenue."))
    review = cl.chat_context.get()[-1]
    control = review.elements[0].props[HITL_CONTROL_PROP]
    fake_gateway.replies += [
        streamed(response(DISPLAY_CHART, id="resp_lg_done")),
        httpx2.Response(200, content=b"png-bytes"),
        streamed(response(message("Saved and plotted."))),
    ]

    result = await chat.on_interrupt_submit(
        cl.Action(
            name=INTERRUPT_ACTION_NAME,
            payload={
                "step_id": control["step_id"],
                "element_id": control["element_id"],
                "revision": control["revision"],
                "outputs": ["approve"],
            },
        )
    )

    assert result == {"ok": True}
    *_, resume, continuation = fake_gateway.bodies("/v1/responses")
    assert resume["previous_response_id"] == "resp_lg_review"
    # The resumed run has finished; the tool result starts a new stateless run.
    assert "previous_response_id" not in continuation
    assert continuation["input"] == [
        {"role": "user", "content": "Plot revenue."},
        DISPLAY_CHART.model_dump(mode="json", exclude_none=True),
        {
            "type": "function_call_output",
            "call_id": "call_chart",
            "output": '{"displayed":true}',
        },
    ]
    assert review.elements == []
    assert transcript()[-1] == "Saved and plotted."
