"""
title: UserValves-Simple / simple-graph
author: langgraph-openai-serve
version: 0.3
description: Static per-user settings for one fixed graph when dynamic per-chat settings are unnecessary.
"""

from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

# This value mirrors the public LGOS wire contract. This standalone Open WebUI
# Function must not import the server package:
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/client_settings.py
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
LGOS_EXTENSION_KEY = "langgraph_openai_serve"
LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "langgraph_openai_serve model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable. Configure the proxy to pass LGOS /v1 requests "
    "and responses through unchanged."
)


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="http://bifrost:8080/openai_passthrough/v1",
            description="OpenAI base URL used for model retrieval and chat.",
        )
        OPENAI_API_KEY: str = Field(
            default="DUMMY",
            description="Bearer token sent to the configured OpenAI endpoint.",
        )
        MODEL: str = Field(
            default="simple-graph",
            min_length=1,
            description="LGOS model exposed by the configured OpenAI endpoint.",
        )
        OPENAI_API_HEADERS: dict[str, str] = Field(
            default_factory=lambda: {"x-model-provider": "lgos-a"},
            description="Additional headers sent with model and chat requests.",
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
        __event_emitter__: Any = None,
    ) -> AsyncIterator[str]:
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])

        try:
            async with self._client() as client:
                model = await self._retrieve_model(client)
                extension = self._model_extension(model)
                if extension is None:
                    await self._emit_limited_functionality_warning(__event_emitter__)
                stream = await client.chat.completions.create(
                    **self._model_request(),
                    messages=messages,
                    metadata=(
                        self._runtime_settings_metadata(__user__)
                        if self._supports_runtime_settings(extension)
                        else {}
                    ),
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
        return {} if encoded == "{}" else {RUNTIME_SETTINGS_METADATA_KEY: encoded}

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=30,
        )

    async def _retrieve_model(self, client: AsyncOpenAI) -> Any:
        try:
            return await client.models.retrieve(**self._model_request())
        except OpenAIError:
            return None

    def _model_request(self) -> dict[str, Any]:
        request: dict[str, Any] = {"model": self.valves.MODEL}
        if self.valves.OPENAI_API_HEADERS:
            request["extra_headers"] = self.valves.OPENAI_API_HEADERS
        return request

    @staticmethod
    def _model_extension(model: Any) -> dict[str, Any] | None:
        extension = (getattr(model, "model_extra", None) or {}).get(LGOS_EXTENSION_KEY)
        if not isinstance(extension, dict) or extension.get("schema_version") != 1:
            return None
        features = extension.get("features")
        if not isinstance(features, list) or any(
            not isinstance(feature, str) for feature in features
        ):
            return None
        return extension

    @staticmethod
    def _supports_runtime_settings(extension: dict[str, Any] | None) -> bool:
        settings = extension.get("client_settings") if extension is not None else None
        return isinstance(settings, dict) and settings.get("schema_version") == 1

    @staticmethod
    async def _emit_limited_functionality_warning(event_emitter: Any) -> None:
        if event_emitter is None:
            return
        await event_emitter(
            {
                "type": "notification",
                "data": {
                    "type": "warning",
                    "content": LIMITED_FUNCTIONALITY_MESSAGE,
                },
            }
        )
