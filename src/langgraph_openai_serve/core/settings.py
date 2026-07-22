import importlib.util
import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_openai_api_prefix(v: str) -> str:
    """Normalize and validate the OpenAI-compatible API mount prefix."""
    if not v.startswith("/"):
        raise ValueError("OPENAI_API_PREFIX must start with '/'.")
    if len(v) > 1:
        normalized = v.rstrip("/")
        if not normalized:
            raise ValueError("OPENAI_API_PREFIX must not contain only slashes.")
        return normalized
    return v


class Settings(BaseSettings):
    """This class is used to load environment variables either from environment or
    from a .env file and store them as class attributes.
    NOTE:
        - environment variables will always take priority over values loaded from a dotenv file
        - environment variable names are case-insensitive
        - environment variable type is inferred from the type hint of the class attribute
        - For environment variables that are not set, a default value should be provided

    For more info, see the related pydantic docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LGOS_",
        env_file_encoding="utf-8",
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
            raise RuntimeError(
                "Langfuse is enabled but the 'langfuse' package is not installed. "
                "Please install it, e.g., with `uv add langgraph-openai-serve[tracing]`."
            )

        required_env_vars = [
            "LANGFUSE_BASE_URL",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]
        missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

        if missing_vars:
            raise RuntimeError(
                "Langfuse is enabled but the following environment variables are not set: "
                f"{', '.join(missing_vars)}. Please set these variables."
            )

        return v


settings = Settings()
