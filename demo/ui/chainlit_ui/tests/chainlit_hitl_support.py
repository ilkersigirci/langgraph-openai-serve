import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion

RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
STATE_TOKEN = "state-token-1"


def recording_message(
    *,
    content: str = "",
    metadata: dict[str, object] | None = None,
    writes: list[tuple[str, Mock, dict[str, object] | None]],
) -> Mock:
    message = Mock(content=content, metadata=metadata)

    async def send() -> Mock:
        writes.append(("send", message, deepcopy(message.metadata)))
        return message

    async def update() -> bool:
        writes.append(("update", message, deepcopy(message.metadata)))
        return True

    message.send = AsyncMock(side_effect=send)
    message.update = AsyncMock(side_effect=update)
    return message


def install_message_factory(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
    *,
    restored: Mock | None = None,
    session: dict[str, object] | None = None,
) -> tuple[
    Mock,
    list[Mock],
    list[tuple[str, Mock, dict[str, object] | None]],
]:
    created: list[Mock] = []
    writes: list[tuple[str, Mock, dict[str, object] | None]] = []

    def create_message(
        content: str = "",
        metadata: dict[str, object] | None = None,
        **_: object,
    ) -> Mock:
        message = recording_message(
            content=content,
            metadata=metadata,
            writes=writes,
        )
        created.append(message)
        return message

    factory = Mock(side_effect=create_message)
    factory.from_dict = Mock(return_value=restored)
    monkeypatch.setattr(hitl.cl, "Message", factory)
    session_data = {} if session is None else session
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: session_data.get(key, default),
            set=lambda key, value: session_data.__setitem__(key, value),
        ),
    )
    return factory, created, writes


def interrupt_call(
    interrupt_id: str,
    payload: object,
    *,
    arguments: object | None = None,
    state_token: str = STATE_TOKEN,
) -> dict[str, object]:
    arguments = (
        {
            "run_id": RUN_ID,
            "state_token": state_token,
            "payload": payload,
        }
        if arguments is None
        else arguments
    )
    return {
        "id": f"lg_interrupt_{interrupt_id}",
        "type": "function",
        "function": {
            "name": "langgraph_interrupt",
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


def completion(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, object]] | None = None,
) -> ChatCompletion:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "interruptible",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


def ledger_step(
    tool_calls: list[dict[str, object]],
    *,
    status: str = "pending",
    model_id: str = "lgos-a/hitl",
    raw_ledger: object | None = None,
) -> dict[str, Any]:
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    if raw_ledger is None:
        ledger_dict: dict[str, object] = {
            "schema_version": 1,
            "status": status,
        }
        if status == "pending":
            ledger_dict.update(
                {
                    "model_id": model_id,
                    "assistant_message": assistant_message,
                }
            )
        ledger: object = ledger_dict
    else:
        ledger = raw_ledger
    return {
        "id": "persisted-ledger-message",
        "type": "assistant_message",
        "name": "Assistant",
        "output": "",
        "createdAt": "2026-08-10T12:00:00Z",
        "metadata": {
            "lgos_chainlit.exclude_from_model_context": True,
            "lgos_chainlit.hitl_interrupt_ledger": ledger,
        },
    }
