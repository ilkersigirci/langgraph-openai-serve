"""Gateway-backed LangChain models shared by the demo graphs."""

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.constants import TAG_NOSTREAM
from pydantic import SecretStr

from lgos_demo_api.core.settings import settings


def chat_completions_model(
    model: str | None = None,
    *,
    private: bool = False,
    **options: Any,
) -> ChatOpenAI:
    """
    Call the Chat Completions model, or another gateway model, on that route.

    LangGraph streams none of a private model's output to the client.
    """
    return ChatOpenAI(
        model=model or settings.OPENAI_CHAT_COMPLETIONS_MODEL,
        base_url=settings.openai_base_url,
        api_key=SecretStr(settings.OPENAI_GATEWAY_API_KEY),
        # ChatOpenAI asks for streamed usage only from OpenAI's default URL; ask
        # through the gateway too, so LGOS can report streamed calls' usage.
        stream_usage=True,
        tags=[TAG_NOSTREAM] if private else None,
        **options,
    )


def responses_model(*, private: bool = False, **options: Any) -> ChatOpenAI:
    """
    Call the Responses model through the gateway without stored responses.

    LangGraph streams none of a private model's output to the client.
    """
    return ChatOpenAI(
        model=settings.OPENAI_RESPONSES_MODEL,
        base_url=settings.openai_base_url,
        api_key=SecretStr(settings.OPENAI_GATEWAY_API_KEY),
        use_responses_api=True,
        store=False,
        tags=[TAG_NOSTREAM] if private else None,
        **options,
    )


def embedding_model() -> OpenAIEmbeddings:
    """Embed text with the configured model through the gateway."""
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        base_url=settings.openai_base_url,
        api_key=SecretStr(settings.OPENAI_GATEWAY_API_KEY),
    )
