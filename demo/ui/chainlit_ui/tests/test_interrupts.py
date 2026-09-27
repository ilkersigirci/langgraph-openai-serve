"""LGOS interrupt payload presentation tests."""

import json

import pytest
from chainlit_utils.chat.human_review import HumanReview
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


@pytest.mark.parametrize(
    ("payload", "review"),
    [
        (
            {
                "question": "Approve refund?",
                "request": "ORDER-123",
                "choices": [" approve ", "reject"],
                "allow_other": True,
            },
            HumanReview(
                prompt="Approve refund?\n\nRequest: ORDER-123",
                choices=("approve", "reject"),
                allow_other=True,
            ),
        ),
        (
            {"question": "Save note?", "filename": "note.md", "choices": [1, 2]},
            HumanReview(prompt='Save note?\n\n{\n  "filename": "note.md"\n}'),
        ),
        ({}, HumanReview(prompt="Human input required.")),
    ],
    ids=["request", "details", "empty"],
)
def test_review_decodes_the_lgos_payload(
    payload: dict[str, object],
    review: HumanReview,
) -> None:
    assert interrupts.interrupt_review(_call(payload)) == review


@pytest.mark.parametrize("payload", [None, [], "approve"])
def test_review_rejects_non_object_payloads(payload: object) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        interrupts.interrupt_review(_call(payload))
