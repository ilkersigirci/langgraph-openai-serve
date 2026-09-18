"""LGOS interrupt review presentation tests."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from lgos_chainlit import interrupts


def _call(payload: object) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="fc_review",
        call_id="call_review",
        name="lgos_interrupt",
        arguments=json.dumps(payload),
        status="completed",
        type="function_call",
    )


def test_review_decodes_the_demo_payload() -> None:
    review = interrupts.InterruptReview.from_call(
        _call(
            {
                "question": "Approve refund?",
                "request": "ORDER-123",
                "choices": ["approve", "reject"],
                "allow_other": True,
            }
        )
    )

    assert review.prompt == "Approve refund?\n\nRequest: ORDER-123"
    assert review.props == {
        "prompt": review.prompt,
        "choices": ["approve", "reject"],
        "allow_other": True,
    }


@pytest.mark.parametrize("payload", [None, [], "approve"])
def test_review_rejects_non_object_payloads(payload: object) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        interrupts.InterruptReview.from_call(_call(payload))


async def test_review_element_returns_a_valid_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = Mock()
    ask = SimpleNamespace(
        content="Approve?",
        send=AsyncMock(return_value={"submitted": True, "resume": " approve "}),
    )
    ask_factory = Mock(return_value=ask)
    reuse = Mock()
    monkeypatch.setattr(interrupts.cl, "CustomElement", Mock(return_value=element))
    monkeypatch.setattr(interrupts.cl, "AskElementMessage", ask_factory)
    monkeypatch.setattr(interrupts, "reuse_persisted_step", reuse)
    ledger = Mock(content="")

    decision = await interrupts.ask_for_interrupt(
        _call({"question": "Approve?", "choices": ["approve", "reject"]}),
        ledger,
    )

    assert decision == "approve"
    reuse.assert_called_once_with(ask, ledger)
    assert ask_factory.call_args.kwargs["element"] is element
    assert ledger.content == "Approve?"
