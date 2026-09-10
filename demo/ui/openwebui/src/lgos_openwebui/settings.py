"""Settings for the Open WebUI synchronization command."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lgos_openwebui.functions.generic.gateway import GatewayRoot, GatewayType


class Settings(BaseSettings):
    """Configuration for the Open WebUI synchronization command."""

    model_config = SettingsConfigDict(
        env_prefix="DEMO_OPENWEBUI_",
        env_ignore_empty=True,
    )

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
    OPENAI_GATEWAY_TYPE: GatewayType = Field(
        validation_alias="OPENAI_GATEWAY_TYPE",
        description="OpenAI gateway used for model synchronization.",
    )
    OPENAI_GATEWAY_BASE_URL: GatewayRoot = Field(
        validation_alias="OPENAI_GATEWAY_BASE_URL",
        description="Gateway root used by the host sync command.",
    )
    OPENAI_GATEWAY_API_KEY: str = Field(
        min_length=1,
        validation_alias="OPENAI_GATEWAY_API_KEY",
        description="API key sent to the configured OpenAI-compatible endpoints.",
    )
