import importlib.util
import os
from typing import TypedDict

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FastAPIDocsKwargs(TypedDict):
    """FastAPI docs kwargs."""

    docs_url: str | None
    redoc_url: str | None
    openapi_url: str | None


def normalize_openai_api_prefix(v: str) -> str:
    """Normalize and validate the OpenAI-compatible API mount prefix."""
    if not v.startswith("/"):
        msg = "OPENAI_API_PREFIX must start with '/'."
        raise ValueError(msg)
    if len(v) > 1:
        normalized = v.rstrip("/")
        if not normalized:
            msg = "OPENAI_API_PREFIX must not contain only slashes."
            raise ValueError(msg)
        return normalized
    return v


class Settings(BaseSettings):
    """Package settings read from explicit values and the process environment."""

    model_config = SettingsConfigDict(
        env_prefix="LGOS_",
        extra="ignore",
    )

    OPENAI_API_PREFIX: str = "/v1"
    OPENAI_API_DOCS_ENABLED: bool = False
    ENABLE_LANGFUSE: bool = False

    @field_validator("OPENAI_API_PREFIX")
    @classmethod
    def validate_openai_api_prefix(cls, v: str) -> str:
        """Validate the mount prefix for OpenAI-compatible endpoints."""
        return normalize_openai_api_prefix(v)

    @field_validator("ENABLE_LANGFUSE")
    @classmethod
    def check_langfuse_settings(cls, v: bool) -> bool:
        """Validate Langfuse settings if enabled."""
        if v is False:
            return v

        if importlib.util.find_spec("langfuse") is None:
            msg = (
                "Langfuse is enabled but the 'langfuse' package is not installed. "
                "Please install it, e.g., with `uv add langgraph-openai-serve[tracing]`."
            )
            raise RuntimeError(msg)

        required_env_vars = [
            "LANGFUSE_BASE_URL",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]
        missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

        if missing_vars:
            msg = (
                "Langfuse is enabled but the following environment variables are not set: "
                f"{', '.join(missing_vars)}. Please set these variables."
            )
            raise RuntimeError(msg)

        return v

    @property
    def fastapi_docs_kwargs(self) -> FastAPIDocsKwargs:
        """Return kwargs to configure FastAPI docs visibility."""
        if self.OPENAI_API_DOCS_ENABLED:
            return {
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "openapi_url": "/openapi.json",
            }
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}


settings = Settings()
