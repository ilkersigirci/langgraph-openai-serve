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
    """A complete OpenAI-compatible endpoint and its credential."""

    base_url: HttpUrlStr
    api_key: str = Field(min_length=1)
    model_routes: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("model_routes")
    @classmethod
    def validate_model_routes(
        cls,
        value: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Keep synthetic route prefixes unambiguous."""
        if any(not route or "/" in route for route in value):
            raise ValueError(
                "OpenAI model routes must be non-empty and contain no '/'."
            )
        return value


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

    OPENAI: OpenAIEndpoint = OpenAIEndpoint(
        base_url="http://localhost:3004/v1",
        api_key="DUMMY",
    )
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
            raise ValueError("CHAINLIT_AUTH_SECRET must be configured.")
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
            raise ValueError(
                f"Configure the required Chainlit OAuth settings: {missing_settings}."
            )
        return self


@cache
def get_chainlit_settings() -> ChainlitSettings:
    """Load and validate the native Chainlit environment once per process."""
    return ChainlitSettings()
