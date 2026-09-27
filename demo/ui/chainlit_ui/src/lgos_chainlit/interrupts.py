"""Present LGOS interrupt payloads in the chainlit-utils review form."""

import json

from chainlit_utils.chat.human_review import HumanReview
from openai.types.responses import ResponseFunctionToolCall

INTERRUPT_ACTION_NAME = "lgos_interrupt_submit"


def interrupt_review(call: ResponseFunctionToolCall) -> HumanReview:
    """Read one LGOS interrupt call as a human-review request."""
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
        and all(isinstance(choice, str) and choice.strip() for choice in raw_choices)
        else ()
    )
    return HumanReview(
        prompt=_prompt(payload),
        choices=choices,
        allow_other=payload.get("allow_other") is True,
    )


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


__all__ = ["INTERRUPT_ACTION_NAME", "interrupt_review"]
