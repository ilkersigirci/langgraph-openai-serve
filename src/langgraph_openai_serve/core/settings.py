import importlib.util
import os
from typing import Annotated

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    PlainValidator,
    TypeAdapter,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)

CustomHttpUrlStr = Annotated[
    str,
    PlainValidator(lambda x: AnyHttpUrlAdapter.validate_strings(x)),
    AfterValidator(lambda x: str(x).rstrip("/")),
]


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

    ENABLE_LANGFUSE: bool = False

    @field_validator("ENABLE_LANGFUSE")
    def check_langfuse_settings(cls, v: bool) -> bool:
        """Validate Langfuse settings if enabled."""
        if v is False:
            return v

        # Check if langfuse package is installed
        if importlib.util.find_spec("langfuse") is None:
            raise RuntimeError(
                "Langfuse is enabled but the 'langfuse' package is not installed. "
                "Please install it, e.g., with `uv add langgraph-openai-serve[tracing]`."
            )

        # Check for required environment variables
        required_env_vars = [
            "LANGFUSE_HOST",
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
