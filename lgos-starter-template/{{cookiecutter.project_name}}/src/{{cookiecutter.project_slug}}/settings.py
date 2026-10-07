"""Graph settings; process environment takes precedence over .env."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # `.env.example` leaves the secret empty; treat empty values as missing.
        env_ignore_empty=True,
        # Validation errors would otherwise print values from `.env`, secrets included.
        hide_input_in_errors=True,
        extra="ignore",
    )

    # Full variable names as aliases make errors name what to set in `.env`.
    OPENAI_BASE_URL: str = Field(validation_alias="APP_OPENAI_BASE_URL")
    OPENAI_API_KEY: SecretStr = Field(validation_alias="APP_OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(validation_alias="APP_OPENAI_MODEL")


settings = Settings()
