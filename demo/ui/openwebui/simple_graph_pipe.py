"""
title: LGOS Simple Graph
author: langgraph-openai-serve
version: 0.1
"""

from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="http://lgos-demo-api:8000/v1",
            description="Base URL for the LangGraph OpenAI-compatible API.",
        )
        OPENAI_API_KEY: str = Field(
            default="DUMMY",
            description="Bearer token sent to the LangGraph API.",
        )

    class UserValves(BaseModel):
        use_history: bool = Field(
            default=False,
            description="Include prior messages in the model input.",
        )
        audience: Literal["general", "beginner", "expert"] = Field(
            default="general",
            description="Adapt explanations for the selected audience.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])
        metadata = self._runtime_settings_metadata(__user__)

        try:
            async with self._client() as client:
                stream = await client.chat.completions.create(
                    model="simple-graph",
                    messages=messages,
                    metadata=metadata,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and (content := chunk.choices[0].delta.content):
                        yield content
        except OpenAIError as exc:
            yield f"Error calling LangGraph API: {exc}"

    def _runtime_settings_metadata(
        self,
        user: dict[str, Any] | None,
    ) -> dict[str, str]:
        settings = (user or {}).get("valves") or self.UserValves()
        encoded = settings.model_dump_json(exclude_defaults=True)
        return {} if encoded == "{}" else {"langgraph_runtime_settings": encoded}

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=30,
        )
