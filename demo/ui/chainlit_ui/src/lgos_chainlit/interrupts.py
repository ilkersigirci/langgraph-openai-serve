"""Render LGOS interrupt payloads as Chainlit review controls."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

INTERRUPT_ACTION_NAME = "lgos_interrupt_submit"
INTERRUPT_ELEMENT_NAME = "InterruptReview"


class InterruptSubmission(BaseModel):
    """The untrusted reference and answers accepted from the browser."""

    model_config = ConfigDict(extra="forbid", strict=True)

    step_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class InterruptReview:
    """The presentation fields understood by the demo review element."""

    prompt: str
    choices: tuple[str, ...]
    allow_other: bool

    @classmethod
    def from_call(cls, call: ResponseFunctionToolCall) -> "InterruptReview":
        try:
            payload = json.loads(call.arguments)
        except (TypeError, ValueError) as exc:
            raise ValueError("Interrupt arguments must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Interrupt arguments must be a JSON object.")

        raw_choices = payload.get("choices")
        choices = (
            tuple(choice.strip() for choice in raw_choices)
            if isinstance(raw_choices, list)
            and raw_choices
            and all(
                isinstance(choice, str) and choice.strip() for choice in raw_choices
            )
            else ()
        )
        return cls(
            prompt=_prompt(payload),
            choices=choices,
            allow_other=not choices or payload.get("allow_other") is True,
        )

    @property
    def props(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "choices": list(self.choices),
            "allow_other": self.allow_other,
        }


def interrupt_review_props(
    calls: Sequence[ResponseFunctionToolCall],
) -> dict[str, object]:
    """Build one custom-element form for the complete interrupt batch."""
    return {"reviews": [InterruptReview.from_call(call).props for call in calls]}


def pending_interrupt_prompt(calls: Sequence[ResponseFunctionToolCall]) -> str:
    """Render the durable text accompanying an interrupt batch."""
    reviews = [InterruptReview.from_call(call) for call in calls]
    if not reviews:
        return ""
    if len(reviews) == 1:
        return reviews[0].prompt
    prompts = "\n\n".join(
        f"{index}. {review.prompt}" for index, review in enumerate(reviews, start=1)
    )
    return f"Human review is required for {len(reviews)} requests.\n\n{prompts}"


def validate_interrupt_outputs(
    calls: Sequence[ResponseFunctionToolCall],
    outputs: Sequence[str],
) -> tuple[str, ...]:
    """Validate all answers against the trusted persisted interrupt calls."""
    if len(outputs) != len(calls):
        raise ValueError("Every interrupt request requires one response.")

    decisions: list[str] = []
    for call, raw_output in zip(calls, outputs, strict=True):
        if not isinstance(raw_output, str) or not (decision := raw_output.strip()):
            raise ValueError("Interrupt responses must be non-empty strings.")
        review = InterruptReview.from_call(call)
        if review.choices and decision not in review.choices and not review.allow_other:
            raise ValueError("An interrupt response is not an allowed choice.")
        decisions.append(decision)
    return tuple(decisions)


def _prompt(payload: dict[str, object]) -> str:
    lines = [str(payload.get("question") or "Human input required.")]
    if payload.get("request"):
        lines.append(f"Request: {payload['request']}")
    else:
        details = {
            key: value
            for key, value in payload.items()
            if key not in {"question", "choices", "allow_other"}
        }
        if details:
            lines.append(json.dumps(details, ensure_ascii=False, indent=2))
    return "\n\n".join(lines)


__all__ = [
    "INTERRUPT_ACTION_NAME",
    "INTERRUPT_ELEMENT_NAME",
    "InterruptReview",
    "InterruptSubmission",
    "interrupt_review_props",
    "pending_interrupt_prompt",
    "validate_interrupt_outputs",
]
