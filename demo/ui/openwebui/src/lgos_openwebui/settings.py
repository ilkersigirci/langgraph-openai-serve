"""Settings for the Open WebUI synchronization command."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the Open WebUI synchronization command."""

    model_config = SettingsConfigDict(env_prefix="DEMO_OPENWEBUI_")

    URL: str = Field(
        default="http://localhost:3003",
        description="Open WebUI API base URL used by the sync command.",
    )
    ADMIN_EMAIL: str = Field(
        default="lgos@example.com",
        description="Email for the Open WebUI account used by the sync command.",
    )
    ADMIN_PASSWORD: str = Field(
        default="lgos",
        description="Password for the Open WebUI account used by the sync command.",
    )
    OPENAI_BASE_URL: str = Field(
        default="http://localhost:3000/openai_passthrough/v1",
        description="OpenAI-compatible base URL used for model metadata.",
    )
    OPENAI_CATALOG_BASE_URL: str = Field(
        default="http://localhost:3000/v1",
        description="OpenAI-compatible base URL used to list the model catalog.",
    )
    API_KEY: str = Field(
        default="DUMMY",
        description="API key sent to the configured OpenAI-compatible endpoints.",
    )
