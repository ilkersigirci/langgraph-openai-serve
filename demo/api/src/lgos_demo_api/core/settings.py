from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    Field,
    PlainValidator,
    TypeAdapter,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)
HttpUrlStr = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings),
    AfterValidator(lambda value: str(value).rstrip("/")),
]


class Settings(BaseSettings):
    """
    Load environment variables either from environment or    from a .env file and store them as class attributes.

    Configuration owned by the standalone demo API.

    Note:
        - environment variables will always take priority over values loaded from a dotenv file
        - environment variable names are case-insensitive
        - environment variable type is inferred from the type hint of the class attribute
        - For environment variables that are not set, a default value should be provided

    For more info, see the related pydantic docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMO_API_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_GATEWAY_BASE_URL: HttpUrlStr = Field(
        default="http://localhost:3000",
        validation_alias="OPENAI_GATEWAY_BASE_URL",
    )
    OPENAI_GATEWAY_API_KEY: str = Field(
        default="DUMMY",
        validation_alias="OPENAI_GATEWAY_API_KEY",
        repr=False,
    )
    OPENAI_CHAT_COMPLETIONS_MODEL: str = "openai/gpt-4.1-mini"
    OPENAI_RESPONSES_MODEL: str = "openai/gpt-6-luna"
    VECTOR_STORE_ID: str | None = None
    OPENAI_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    WEB_SEARCH_BACKEND: Literal["http", "openai"] = "http"
    WEB_SEARCH_URL: HttpUrlStr = "https://searxng.example.com/search"
    FILES_BASE_URL: HttpUrlStr = "http://localhost:3006/v1"

    @property
    def openai_base_url(self) -> str:
        """OpenAI model routes shared by the bundled gateways."""
        return f"{self.OPENAI_GATEWAY_BASE_URL}/v1"

    @property
    def vector_store_base_url(self) -> str:
        """Keep knowledge Files and vector stores in the upstream account."""
        return f"{self.OPENAI_GATEWAY_BASE_URL}/openai_passthrough/v1"


settings = Settings()
