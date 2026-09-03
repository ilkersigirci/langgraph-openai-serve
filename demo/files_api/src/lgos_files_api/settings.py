"""Environment-backed settings for the Files service."""

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
    """Configuration owned solely by the standalone Files service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMO_API_FILES_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PORT: int = 8000
    BUCKET: str | None = None
    S3_ENDPOINT: HttpUrlStr | None = None
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_DEFAULT_REGION: str | None = None


settings = Settings()
