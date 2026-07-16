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


class Settings(BaseSettings):
    """Configuration owned by the demo API and UI applications."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMO_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        extra="ignore",
    )

    POSTGRES_URI: str = "postgresql://lgos:lgos@localhost:5432/lgos"
    OPENAI_BASE_URL: HttpUrlStr = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = "DUMMY"
    OPENAI_MODEL: str = "gpt-5.4-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAINLIT_INFERENCE: OpenAIEndpoint = OpenAIEndpoint(
        base_url="http://localhost:8000/v1",
        api_key="DUMMY",
    )
    CHAINLIT_INFERENCE_MODEL_PREFIX: str = ""
    CHAINLIT_DISCOVERY: OpenAIEndpoint | None = None
    CHAINLIT_HITL_MODEL: str = "interruptible-approval"
    CHAINLIT_UI_FILE: Literal["simple", "hitl"] = "simple"
    CHAINLIT_LOGIN_TYPE: ChainlitLoginType = "mock"

    @property
    def chainlit_discovery_endpoint(self) -> OpenAIEndpoint:
        """Use a complete discovery endpoint, or the inference endpoint."""
        return self.CHAINLIT_DISCOVERY or self.CHAINLIT_INFERENCE

    def chainlit_inference_model(self, model: str) -> str:
        """Apply an optional proxy provider/model namespace."""
        return f"{self.CHAINLIT_INFERENCE_MODEL_PREFIX}{model}"


settings = Settings()


class ChainlitSettings(BaseSettings):
    """Validated native settings for the persistent Chainlit demo."""

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
        if settings.CHAINLIT_LOGIN_TYPE != "oauth":
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
    """Load and validate the Chainlit environment once per process."""
    return ChainlitSettings()
