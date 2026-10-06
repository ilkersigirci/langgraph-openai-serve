"""Graph settings; process environment takes precedence over .env."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: SecretStr = SecretStr("DUMMY")
    OPENAI_MODEL: str = "gpt-5.4-mini"


settings = Settings()
