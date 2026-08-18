"""
title: UserValves-Simple / simple-graph

author: langgraph-openai-serve
version: 0.5
description: Static per-user settings for one fixed graph when dynamic per-chat settings are unnecessary.
"""

import os
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
            default=os.environ.get(
                "OPENAI_API_BASE_URL",
                "http://lgos-bifrost:8080/openai_passthrough/v1",
            ),
            description="OpenAI-compatible base URL used for retrieval and chat.",
        )
        OPENAI_API_KEY: str = Field(
            default=os.environ.get("OPENAI_API_KEY", "DUMMY"),
            description="API key sent to the configured OpenAI-compatible endpoint.",
            json_schema_extra={"input": {"type": "password"}},
        )
        MODEL: str = Field(
            default="lgos-a/simple-graph",
            min_length=1,
            description="Bifrost provider-qualified LGOS model ID.",
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
        except ValueError as exc:
            yield str(exc)
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
        provider, separator, model = self.valves.MODEL.partition("/")
        if not provider or not separator or not model:
            msg = (
                "Bifrost model ID must use the provider/model format: "
                f"{self.valves.MODEL!r}."
            )
            raise ValueError(msg)
        return {
            "model": model,
            "extra_headers": {"x-model-provider": provider},
        }

    @staticmethod
    def _model_extension(model: Any) -> dict[str, Any] | None:
        extension = (getattr(model, "model_extra", None) or {}).get(LGOS_EXTENSION_KEY)
        if not isinstance(extension, dict) or extension.get("schema_version") != 1:
            return None
        description = extension.get("description")
        features = extension.get("features")
        if (
            not isinstance(features, list)
            or any(not isinstance(feature, str) for feature in features)
            or not isinstance(description, str)
            or not description.strip()
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
