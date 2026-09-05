"""
title: UserValves Simple
author: langgraph-openai-serve
version: 0.7
description: Static per-user runtime settings for the simple-graph example.
"""

from typing import Any, Literal, cast

from pydantic import BaseModel, Field


class Filter:
    class UserValves(BaseModel):
        use_history: bool = Field(
            default=False,
            description="Include prior messages in the model input.",
        )
        audience: Literal["general", "beginner", "expert"] = Field(
            default="general",
            description="Adapt explanations for the selected audience.",
        )

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __metadata__: dict[str, Any],
    ) -> dict[str, Any]:
        settings = cast("Filter.UserValves", __user__["valves"])
        # The shared Pipe encodes these values as OpenAI runtime metadata.
        __metadata__["chat_variables"] = settings.model_dump()
        return body
