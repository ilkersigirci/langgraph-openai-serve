"""Gateway-backed LangChain models shared by the demo graphs."""

from typing import Any

import httpx2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.config import get_config
from langgraph.constants import TAG_NOSTREAM
from openai import DefaultAsyncHttpxClient
from pydantic import SecretStr

from lgos_demo_api.core.settings import settings

# LiteLLM's standard header: it bills the call to this end user, while the
# graph API's key stays the caller.
END_USER_HEADER = "x-litellm-end-user-id"


async def attribute_end_user(request: httpx2.Request) -> None:  # ruff: ignore[unused-async] - httpx requires async event hooks on an AsyncClient.
    """Bill a gateway call made during a graph run to that run's end user."""
    try:
        metadata = get_config().get("metadata", {})
    except RuntimeError:
        return  # Not inside a graph run.
    # LGOS records each run's end user here, whether or not Langfuse is on.
    if user := metadata.get("lgos.user"):
        request.headers[END_USER_HEADER] = user


# The one HTTP client for gateway calls: the OpenAI SDK's default with the hook.
# Like LangChain's own cached default client, it lives as long as the process.
GATEWAY_HTTP_CLIENT = DefaultAsyncHttpxClient(
    event_hooks={"request": [attribute_end_user]}
)


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
        http_async_client=GATEWAY_HTTP_CLIENT,
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
        http_async_client=GATEWAY_HTTP_CLIENT,
        **options,
    )


def embedding_model() -> OpenAIEmbeddings:
    """Embed text with the configured model through the gateway."""
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        base_url=settings.openai_base_url,
        api_key=SecretStr(settings.OPENAI_GATEWAY_API_KEY),
        http_async_client=GATEWAY_HTTP_CLIENT,
    )
