from collections.abc import AsyncGenerator

import httpx2
from langchain_core.language_models import BaseChatModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent
from openai_codex.generated.notification_registry import NOTIFICATION_MODELS
from openai_codex.models import Notification

from lgos_api_coding_agent.app import create_app
from lgos_api_coding_agent.codex_model import CodexChatModel
from lgos_api_coding_agent.codex_runtime import CodexTurn


def openai_client(model: BaseChatModel) -> AsyncOpenAI:
    """Return an OpenAI client for the app serving ``model`` in-process."""
    return AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=create_app(model))
        ),
    )


def answer_deltas(events: list[ResponseStreamEvent]) -> list[str]:
    phases = {
        event.item.id: event.item.phase
        for event in events
        if event.type == "response.output_item.added" and event.item.type == "message"
    }
    return [
        event.delta
        for event in events
        if event.type == "response.output_text.delta"
        and phases.get(event.item_id) == "final_answer"
    ]


def event(method: str, **payload: object) -> Notification:
    return Notification(
        method,
        NOTIFICATION_MODELS[method].model_validate(
            {
                "threadId": "thread",
                "turnId": "turn",
                "startedAtMs": 0,
                "completedAtMs": 1,
                **payload,
            }
        ),
    )


def message(kind: str, identifier: str, text: str, phase: str | None) -> Notification:
    return event(
        f"item/{kind}",
        item={"type": "agentMessage", "id": identifier, "text": text, "phase": phase},
    )


def terminal(status: str = "completed") -> Notification:
    return event("turn/completed", turn={"id": "turn", "items": [], "status": status})


def model(
    events: list[Notification | str], turns: list[CodexTurn] | None = None
) -> CodexChatModel:
    async def source(turn: CodexTurn) -> AsyncGenerator[Notification | str, None]:
        if turns is not None:
            turns.append(turn)
        for notification in events:
            yield notification

    return CodexChatModel(event_source=source, model_name="fixture")
