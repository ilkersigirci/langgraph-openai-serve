from functools import cache
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    Field,
    PlainValidator,
    PostgresDsn,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from lgos_chainlit.gateway import GatewayType

AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)
HttpUrlStr = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings),
    AfterValidator(lambda value: str(value).rstrip("/")),
]
ChainlitLoginType = Literal["mock", "oauth"]
PLACEHOLDER = "TO_BE_FILLED"
REQUIRED_OAUTH_SETTINGS = (
    "OAUTH_GENERIC_CLIENT_ID",
    "OAUTH_GENERIC_CLIENT_SECRET",
    "OAUTH_GENERIC_AUTH_URL",
    "OAUTH_GENERIC_TOKEN_URL",
    "OAUTH_GENERIC_USER_INFO_URL",
    "OAUTH_GENERIC_SCOPES",
)


def _is_unconfigured(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip() == PLACEHOLDER


class OpenAIEndpoint(BaseModel):
    """OpenAI-compatible endpoints and their shared credential."""

    gateway_base_url: HttpUrlStr | None = Field(
        default=None,
        description=(
            "Optional gateway root override. The demo selects the local root from "
            "OPENAI_GATEWAY_TYPE when this is unset."
        ),
    )
    api_key: str = Field(
        min_length=1,
        description="API key sent to the configured OpenAI-compatible endpoints.",
    )


class Settings(BaseSettings):
    """Configuration owned by the standalone Chainlit application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMO_CHAINLIT_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        extra="ignore",
    )

    OPENAI_GATEWAY_TYPE: GatewayType = Field(
        default="litellm",
        validation_alias="OPENAI_GATEWAY_TYPE",
        description="OpenAI gateway used by every Chainlit OpenAI client.",
    )
    OPENAI: OpenAIEndpoint = OpenAIEndpoint(api_key="sk-lgos-litellm-demo")
    HITL_MODEL: str = "interruptible-approval"
    UI_FILE: Literal["simple", "hitl"] = "simple"
    LOGIN_TYPE: ChainlitLoginType = "mock"


settings = Settings()


class ChainlitSettings(BaseSettings):
    """Validate settings consumed natively by persistent Chainlit."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    DATABASE_URL: PostgresDsn
    CHAINLIT_AUTH_SECRET: str
    BUCKET_NAME: str
    APP_AWS_ACCESS_KEY: str
    APP_AWS_SECRET_KEY: str
    APP_AWS_REGION: str
    DEV_AWS_ENDPOINT: HttpUrlStr
    OAUTH_GENERIC_CLIENT_ID: str | None = None
    OAUTH_GENERIC_CLIENT_SECRET: str | None = None
    OAUTH_GENERIC_AUTH_URL: str | None = None
    OAUTH_GENERIC_TOKEN_URL: str | None = None
    OAUTH_GENERIC_USER_INFO_URL: str | None = None
    OAUTH_GENERIC_SCOPES: str | None = None

    @field_validator("CHAINLIT_AUTH_SECRET")
    @classmethod
    def validate_auth_secret(cls, value: str) -> str:
        """Reject a missing or example signing secret before Chainlit starts."""
        if _is_unconfigured(value):
            msg = "CHAINLIT_AUTH_SECRET must be configured."
            raise ValueError(msg)
        return value

    @field_validator(
        "BUCKET_NAME",
        "APP_AWS_ACCESS_KEY",
        "APP_AWS_SECRET_KEY",
        "APP_AWS_REGION",
    )
    @classmethod
    def validate_s3_setting(cls, value: str) -> str:
        """Reject incomplete native Chainlit S3 configuration."""
        if _is_unconfigured(value):
            msg = "Chainlit S3 storage must be configured."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_oauth_settings(self) -> Self:
        """Require the generic-provider fields only when OAuth is selected."""
        if settings.LOGIN_TYPE != "oauth":
            return self

        missing = [
            name
            for name in REQUIRED_OAUTH_SETTINGS
            if _is_unconfigured(getattr(self, name))
        ]
        if missing:
            missing_settings = ", ".join(missing)
            msg = f"Configure the required Chainlit OAuth settings: {missing_settings}."
            raise ValueError(msg)
        return self


@cache
def get_chainlit_settings() -> ChainlitSettings:
    """Load and validate the native Chainlit environment once per process."""
    return ChainlitSettings()
