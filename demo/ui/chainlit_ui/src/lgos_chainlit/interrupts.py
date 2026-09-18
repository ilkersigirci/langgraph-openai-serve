"""Render LGOS interrupt payloads as Chainlit review controls."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

import chainlit as cl
from chainlit_utils.chat.history import send_ui_message
from chainlit_utils.chat.resume import reuse_persisted_step
from openai.types.responses import ResponseFunctionToolCall

INTERRUPT_ELEMENT_NAME = "InterruptReview"


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
            tuple(raw_choices)
            if isinstance(raw_choices, list)
            and raw_choices
            and all(isinstance(choice, str) and choice for choice in raw_choices)
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


def pending_interrupt_prompt(calls: Sequence[ResponseFunctionToolCall]) -> str:
    """Render a durable ledger message without skipping malformed calls."""
    if not calls:
        return ""
    try:
        return InterruptReview.from_call(calls[0]).prompt
    except ValueError:
        return ""


async def ask_for_interrupt(
    call: ResponseFunctionToolCall,
    ledger_message: cl.Message,
) -> str | None:
    """Ask for one interrupt output using the demo's review element."""
    try:
        review = InterruptReview.from_call(call)
    except ValueError:
        await send_ui_message("Received an unsupported interrupt payload.")
        return None

    message = cl.AskElementMessage(
        content=review.prompt,
        element=cl.CustomElement(
            name=INTERRUPT_ELEMENT_NAME,
            display="inline",
            props=review.props,
        ),
        timeout=300,
    )
    # Chainlit persists the ask message without its live controls. Updating the
    # ledger step avoids adding another message whenever a thread reconnects.
    reuse_persisted_step(message, ledger_message)
    ledger_message.content = message.content
    response = await message.send()
    ledger_message.content = message.content

    if not response:
        await send_ui_message("Interrupt input timed out.")
        return None
    if not isinstance(response, dict) or response.get("submitted") is not True:
        await send_ui_message("Interrupt was cancelled.")
        return None

    raw_decision = response.get("resume")
    if not isinstance(raw_decision, str) or not raw_decision.strip():
        await send_ui_message("No interrupt response was received.")
        return None
    decision = raw_decision.strip()
    if review.choices and decision not in review.choices and not review.allow_other:
        await send_ui_message("No interrupt response was received.")
        return None
    return decision


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
    "INTERRUPT_ELEMENT_NAME",
    "InterruptReview",
    "ask_for_interrupt",
    "pending_interrupt_prompt",
]
