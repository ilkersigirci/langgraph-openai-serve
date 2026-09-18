"""LGOS interrupt review presentation and validation tests."""

import json

import pytest
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError

from lgos_chainlit import interrupts


def _call(payload: object, *, suffix: str = "review") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        call_id=f"call_{suffix}",
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
                "choices": [" approve ", "reject"],
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


def test_review_props_and_prompt_cover_the_complete_batch() -> None:
    calls = [
        _call(
            {"question": "Approve refund?", "choices": ["approve", "reject"]},
            suffix="refund",
        ),
        _call({"question": "Choose carrier"}, suffix="carrier"),
    ]

    props = interrupts.interrupt_review_props(calls)
    prompt = interrupts.pending_interrupt_prompt(calls)

    assert props == {
        "reviews": [
            {
                "prompt": "Approve refund?",
                "choices": ["approve", "reject"],
                "allow_other": False,
            },
            {
                "prompt": "Choose carrier",
                "choices": [],
                "allow_other": True,
            },
        ]
    }
    assert prompt == (
        "Human review is required for 2 requests.\n\n"
        "1. Approve refund?\n\n2. Choose carrier"
    )


def test_interrupt_outputs_are_validated_against_trusted_calls() -> None:
    calls = [
        _call(
            {"question": "Approve?", "choices": [" approve ", "reject"]},
            suffix="approval",
        ),
        _call({"question": "Explain"}, suffix="explanation"),
    ]

    assert interrupts.validate_interrupt_outputs(
        calls,
        [" approve ", " because it is safe "],
    ) == ("approve", "because it is safe")


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (["approve"], "Every interrupt request"),
        (["approve", "  "], "non-empty strings"),
        (["forged", "reason"], "not an allowed choice"),
    ],
)
def test_invalid_interrupt_outputs_are_rejected(
    outputs: list[str],
    message: str,
) -> None:
    calls = [
        _call(
            {"question": "Approve?", "choices": ["approve", "reject"]},
            suffix="approval",
        ),
        _call({"question": "Explain"}, suffix="explanation"),
    ]

    with pytest.raises(ValueError, match=message):
        interrupts.validate_interrupt_outputs(calls, outputs)


def test_submission_schema_accepts_only_the_action_contract() -> None:
    submission = interrupts.InterruptSubmission.model_validate(
        {
            "step_id": "step-1",
            "element_id": "element-1",
            "revision": "resp-1",
            "outputs": ["approve"],
        }
    )

    assert submission.outputs == ["approve"]
    with pytest.raises(ValidationError):
        interrupts.InterruptSubmission.model_validate(
            {
                "step_id": "step-1",
                "element_id": "element-1",
                "revision": "resp-1",
                "outputs": ["approve"],
                "model_id": "browser-controlled",
            }
        )
    with pytest.raises(ValidationError):
        interrupts.InterruptSubmission.model_validate(
            {
                "step_id": "step-1",
                "element_id": "element-1",
                "revision": "resp-1",
                "outputs": "approve",
            }
        )
