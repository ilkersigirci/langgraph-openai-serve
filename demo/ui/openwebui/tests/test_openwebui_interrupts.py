import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lgos_openwebui.functions import generic
from lgos_openwebui.functions.generic import Pipe
from lgos_openwebui.functions.generic import pipe as generic_pipe

from .openwebui_support import (
    INTERRUPT_PAYLOAD,
    USER_REQUEST,
    ScriptedChat,
    body,
    collect_response,
    completion,
    interrupt_call,
    interrupt_response,
    stream_chunk,
)


async def _adapted_ask_user_call(
    monkeypatch: pytest.MonkeyPatch,
    pipe: Pipe,
    *calls: dict[str, Any],
) -> dict[str, Any]:
    monkeypatch.setattr(
        generic_pipe,
        "_chat",
        ScriptedChat(((), completion(tool_calls=list(calls)))),
    )
    chunks = await collect_response(pipe.pipe(body=body(USER_REQUEST)))
    chunk = chunks[0]
    assert isinstance(chunk, dict)
    return chunk["choices"][0]["delta"]["tool_calls"][0]


def _answer_message(
    ask_call: dict[str, Any],
    answers: dict[str, Any],
    *,
    status: str = "answered",
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": ask_call["id"],
        "content": json.dumps({"status": status, "answers": answers}),
    }


def _option(index: int) -> dict[str, Any]:
    return {"type": "option", "option_index": index}


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "complete"])
async def test_pipe_adapts_interrupt_to_persisted_ask_user(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    *,
    stream: bool,
) -> None:
    response = interrupt_response()
    if stream:
        monkeypatch.setattr(generic_pipe, "_chat", ScriptedChat(((), response)))
    else:
        monkeypatch.setattr(
            generic_pipe, "_chat_completion", AsyncMock(return_value=response)
        )

    chunks = await collect_response(
        configured_pipe.pipe(body=body(USER_REQUEST, stream=stream))
    )

    first_chunk = chunks[0]
    assert isinstance(first_chunk, dict)
    choice = first_chunk["choices"][0]
    message = choice["delta" if stream else "message"]
    native_call = message["tool_calls"][0]
    if stream:
        assert native_call["index"] == 0
    assert native_call["function"]["name"] == "ask_user"

    arguments = json.loads(native_call["function"]["arguments"])
    assert arguments["allow_other"] is True
    question = arguments["questions"][0]
    assert question["id"] == "resume_0"
    assert question["question"].startswith("How should the refund be handled?")
    assert json.loads(question["question"].split("\n\n", 1)[1]) == {
        "action": "refund",
        "request": USER_REQUEST,
    }
    assert [option["label"] for option in question["options"]] == [
        "approve",
        "reject",
    ]
    last_chunk = chunks[-1]
    assert isinstance(last_chunk, dict)
    finish = last_chunk["choices"][0] if stream else choice
    assert finish["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    ("answer", "resume"),
    [
        pytest.param(_option(0), "approve", id="choice"),
        pytest.param(
            {"type": "other", "text": "  Error: check the address first.  "},
            "Error: check the address first.",
            id="custom-text",
        ),
    ],
)
async def test_pipe_translates_persisted_answer_to_lgos_resume(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    answer: dict[str, Any],
    resume: str,
) -> None:
    interrupt = interrupt_call("interrupt-1", INTERRUPT_PAYLOAD)
    ask_call = await _adapted_ask_user_call(
        monkeypatch,
        configured_pipe,
        interrupt,
    )
    messages = [
        {"role": "user", "content": USER_REQUEST},
        {"role": "assistant", "content": "", "tool_calls": [ask_call]},
        _answer_message(ask_call, {"resume_0": answer}),
    ]
    chat = ScriptedChat((("Finished.",), completion("Finished.")))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    chunks = await collect_response(
        configured_pipe.pipe(body={**body(USER_REQUEST), "messages": messages})
    )

    assert chunks == [
        stream_chunk(content="Finished.").model_dump(
            mode="json",
            exclude_none=True,
        )
    ]
    assert chat.calls[0][0] == [
        {"role": "assistant", "content": None, "tool_calls": [interrupt]},
        {
            "role": "tool",
            "tool_call_id": interrupt["id"],
            "content": json.dumps({"resume": resume}),
        },
    ]


async def test_pipe_rebuilds_complete_interrupt_batch_in_call_order(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    first = interrupt_call(
        "interrupt-1",
        {**INTERRUPT_PAYLOAD, "question": "Review refund?"},
    )
    second = interrupt_call(
        "interrupt-2",
        {
            **INTERRUPT_PAYLOAD,
            "question": "Review notification?",
            "request": "Email the customer",
        },
    )
    ask_call = await _adapted_ask_user_call(
        monkeypatch,
        configured_pipe,
        first,
        second,
    )
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [ask_call]},
        _answer_message(
            ask_call,
            {"resume_1": _option(1), "resume_0": _option(0)},
        ),
    ]
    chat = ScriptedChat((("Applied.",), completion("Applied.")))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    await collect_response(
        configured_pipe.pipe(body={**body(USER_REQUEST), "messages": messages})
    )

    assert chat.calls[0][0] == [
        {"role": "assistant", "content": None, "tool_calls": [first, second]},
        {
            "role": "tool",
            "tool_call_id": first["id"],
            "content": '{"resume": "approve"}',
        },
        {
            "role": "tool",
            "tool_call_id": second["id"],
            "content": '{"resume": "reject"}',
        },
    ]


async def test_pipe_leaves_ordinary_ask_user_ledger_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-other",
                    "type": "function",
                    "function": {"name": "ask_user", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-other", "content": "answer"},
    ]
    chat = ScriptedChat((("Done.",), completion("Done.")))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    await collect_response(
        configured_pipe.pipe(body={**body(USER_REQUEST), "messages": messages})
    )

    assert chat.calls[0][0] == messages


async def test_pipe_ignores_completed_interrupts_from_older_turns(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    ask_call = await _adapted_ask_user_call(
        monkeypatch,
        configured_pipe,
        interrupt_call("interrupt-1", INTERRUPT_PAYLOAD),
    )
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [ask_call]},
        _answer_message(ask_call, {"resume_0": _option(0)}),
        {"role": "assistant", "content": "Finished."},
        {"role": "user", "content": "What happened next?"},
    ]
    chat = ScriptedChat((("Next answer.",), completion("Next answer.")))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    await collect_response(
        configured_pipe.pipe(body={**body(USER_REQUEST), "messages": messages})
    )

    assert chat.calls[0][0] == messages


async def test_pipe_rejects_incomplete_persisted_answer(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    ask_call = await _adapted_ask_user_call(
        monkeypatch,
        configured_pipe,
        interrupt_call("interrupt-1", INTERRUPT_PAYLOAD),
    )
    messages = [{"role": "assistant", "content": "", "tool_calls": [ask_call]}]

    chunks = await collect_response(
        configured_pipe.pipe(body={**body(USER_REQUEST), "messages": messages})
    )

    assert chunks == [
        {"error": {"detail": "Open WebUI returned an incomplete interrupt batch."}}
    ]


async def test_pipe_rejects_invalid_answer(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    ask_call = await _adapted_ask_user_call(
        monkeypatch,
        configured_pipe,
        interrupt_call("interrupt-1", INTERRUPT_PAYLOAD),
    )
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [ask_call]},
        {
            "role": "tool",
            "tool_call_id": ask_call["id"],
            "content": "unexpected",
        },
    ]

    chunks = await collect_response(
        configured_pipe.pipe(body={**body(USER_REQUEST), "messages": messages})
    )

    assert chunks == [
        {"error": {"detail": "Open WebUI returned an invalid interrupt answer."}}
    ]


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "complete"])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param(generic.ASK_USER_REJECTED_OUTPUT, id="rejected"),
        pytest.param(
            json.dumps({"status": "cancelled", "answers": {}}),
            id="timed-out",
        ),
    ],
)
async def test_pipe_ends_turn_without_resuming_cancelled_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    content: str,
    *,
    stream: bool,
) -> None:
    ask_call = await _adapted_ask_user_call(
        monkeypatch,
        configured_pipe,
        interrupt_call("interrupt-1", INTERRUPT_PAYLOAD),
    )
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [ask_call]},
        {"role": "tool", "tool_call_id": ask_call["id"], "content": content},
    ]

    chunks = await collect_response(
        configured_pipe.pipe(
            body={**body(USER_REQUEST, stream=stream), "messages": messages}
        )
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], dict)
    choice = chunks[0]["choices"][0]
    assert choice["finish_reason"] == "stop"
    response = choice["delta" if stream else "message"]
    assert response["content"] == generic.INTERRUPT_CANCELLED_MESSAGE


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        pytest.param(
            "Question",
            "Open WebUI requires an object interrupt payload.",
            id="non-object",
        ),
        pytest.param(
            {"choices": ["yes", "no"]},
            "Open WebUI interrupt payload requires a question.",
            id="missing-question",
        ),
        pytest.param(
            {"question": "Proceed?", "choices": ["yes"]},
            "Open WebUI interrupts require 2-3 unique string choices.",
            id="too-few-choices",
        ),
        pytest.param(
            {
                "question": "Proceed?",
                "choices": ["yes", "no"],
                "context": "x" * 500,
            },
            "Open WebUI interrupt question exceeds 500 characters.",
            id="rendered-question-too-long",
        ),
    ],
)
async def test_pipe_rejects_payloads_unsupported_by_native_ask_user(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    payload: object,
    detail: str,
) -> None:
    chat = ScriptedChat(((), completion(tool_calls=[interrupt_call("i", payload)])))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    chunks = await collect_response(configured_pipe.pipe(body=body(USER_REQUEST)))

    assert chunks == [{"error": {"detail": detail}}]


async def test_pipe_rejects_more_than_three_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    calls = [interrupt_call(str(index), INTERRUPT_PAYLOAD) for index in range(4)]
    chat = ScriptedChat(((), completion(tool_calls=calls)))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    chunks = await collect_response(configured_pipe.pipe(body=body(USER_REQUEST)))

    assert chunks == [
        {"error": {"detail": "Open WebUI supports at most 3 interrupts per batch."}}
    ]


async def test_pipe_leaves_model_generated_ask_user_call_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    ask_user_call = {
        "id": "call-question",
        "type": "function",
        "function": {
            "name": "ask_user",
            "arguments": json.dumps(
                {
                    "questions": [
                        {
                            "id": "format",
                            "header": "Format",
                            "question": "Which format should I use?",
                            "options": [
                                {"label": "Short", "description": "Be concise."},
                                {"label": "Long", "description": "Add detail."},
                            ],
                        }
                    ]
                }
            ),
        },
    }
    chat = ScriptedChat(((), completion(tool_calls=[ask_user_call])))
    monkeypatch.setattr(generic_pipe, "_chat", chat)

    chunks = await collect_response(configured_pipe.pipe(body=body(USER_REQUEST)))

    chunk = chunks[0]
    assert isinstance(chunk, dict)
    assert chunk["choices"][0]["delta"]["tool_calls"] == [{"index": 0, **ask_user_call}]
