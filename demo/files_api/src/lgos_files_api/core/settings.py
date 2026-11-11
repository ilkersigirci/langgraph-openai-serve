"""Environment-backed settings for the Files service."""

from typing import Annotated

from pydantic import AfterValidator, AnyHttpUrl, Field, PlainValidator, TypeAdapter
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
        # Compose passes unset variables through as empty strings.
        env_ignore_empty=True,
        extra="ignore",
    )

    PORT: int = 8000
    BUCKET: str | None = None
    S3_ENDPOINT: HttpUrlStr | None = None
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_DEFAULT_REGION: str | None = None
    # OpenAI accepts uploads up to exactly 512 MiB.
    MAX_UPLOAD_BYTES: int = Field(default=512 * 1024 * 1024, gt=0)
    # Extractions beyond the limit wait for a slot.
    MAX_CONCURRENT_EXTRACTIONS: int = Field(default=2, gt=0)
    # xberg sizes its thread pools by the host's cores, not the container's CPU
    # limit, and its memory grows with them.
    EXTRACTION_THREADS: int = Field(default=1, gt=0)
    # Open WebUI waits 300 seconds for a document's text, counted from the
    # upload; this limit starts once the extraction has a slot.
    EXTRACTION_TIMEOUT_SECONDS: int = Field(default=240, gt=0)


settings = Settings()
