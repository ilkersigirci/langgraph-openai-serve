from typing import Annotated

from pydantic import AfterValidator, AnyHttpUrl, PlainValidator, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)
HttpUrlStr = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings),
    AfterValidator(lambda value: str(value).rstrip("/")),
]


class Settings(BaseSettings):
    """Configuration owned by the demo API application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMO_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    POSTGRES_URI: str = "postgresql://lgos:lgos@localhost:5432/lgos"
    OPENAI_BASE_URL: HttpUrlStr = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = "DUMMY"
    OPENAI_MODEL: str = "gpt-5.4-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAINLIT_OPENAI_BASE_URL: HttpUrlStr = "http://localhost:8000/v1"
    CHAINLIT_HITL_MODEL: str = "interruptible-approval"
    CHAINLIT_UI_FILE: str = "simple"


settings = Settings()
