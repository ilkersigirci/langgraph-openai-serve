import httpx2
from langchain_core.language_models import BaseChatModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent

from lgos_api_coding_agent.app import create_app


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
