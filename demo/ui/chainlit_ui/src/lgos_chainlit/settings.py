from functools import cache
from typing import Annotated, Literal, Self

from cryptography.fernet import Fernet
from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    AnyUrl,
    Field,
    PlainValidator,
    PostgresDsn,
    SecretStr,
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
    "OAUTH_GENERIC_SCOPES",
    "CHAINLIT_URL",
)


def _is_unconfigured(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip() == PLACEHOLDER


class Settings(BaseSettings):
    """Configuration owned by the standalone Chainlit application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMO_CHAINLIT_",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        extra="ignore",
    )

    OPENAI_GATEWAY_TYPE: GatewayType = Field(
        validation_alias="OPENAI_GATEWAY_TYPE",
        description="OpenAI gateway used by every Chainlit OpenAI client.",
    )
    OPENAI_GATEWAY_BASE_URL: HttpUrlStr = Field(
        validation_alias="OPENAI_GATEWAY_BASE_URL",
        description="Gateway root without the OpenAI API path.",
    )
    OPENAI_GATEWAY_API_KEY: str | None = Field(
        default=None,
        min_length=1,
        repr=False,
        validation_alias="OPENAI_GATEWAY_API_KEY",
        description="Shared gateway API key for mock login; ignored in OAuth mode.",
    )
    HITL_MODEL: str = "interruptible-approval"
    UI_FILE: Literal["simple", "hitl"] = "simple"
    LOGIN_TYPE: ChainlitLoginType = "mock"
    OAUTH_RESOURCE: str | None = Field(default=None, min_length=1)
    OAUTH_ISSUER: str | None = None
    OAUTH_CLIENT_AUTH_METHOD: Literal["client_secret_basic", "client_secret_post"] = (
        "client_secret_basic"
    )
    OAUTH_ENCRYPTION_KEYS: list[SecretStr] = Field(default_factory=list, repr=False)

    @field_validator("OAUTH_RESOURCE")
    @classmethod
    def validate_resource(cls, value: str | None) -> str | None:
        if value is not None:
            resource = TypeAdapter(AnyUrl).validate_python(value)
            if resource.fragment is not None or value != value.strip():
                raise ValueError(
                    "OAuth resource must be an absolute URI without a fragment or surrounding whitespace."
                )
        return value

    @field_validator("OAUTH_ISSUER")
    @classmethod
    def validate_issuer(cls, value: str | None) -> str | None:
        if value is not None:
            issuer = AnyHttpUrlAdapter.validate_python(value)
            if (
                issuer.scheme != "https"
                or issuer.query is not None
                or issuer.fragment is not None
                or issuer.username
                or value != value.strip()
            ):
                raise ValueError(
                    "OAuth issuer must be an HTTPS URL without credentials, query, or fragment."
                )
        return value

    @field_validator("OAUTH_ENCRYPTION_KEYS")
    @classmethod
    def validate_encryption_keys(cls, keys: list[SecretStr]) -> list[SecretStr]:
        for key in keys:
            try:
                Fernet(key.get_secret_value())
            except (ValueError, TypeError):
                raise ValueError("OAuth encryption keys must be Fernet keys.") from None
        return keys

    @model_validator(mode="after")
    def validate_gateway_auth(self) -> Self:
        if self.LOGIN_TYPE == "oauth":
            if not self.OAUTH_ISSUER:
                raise ValueError(
                    "DEMO_CHAINLIT_OAUTH_ISSUER must be an HTTPS issuer URL."
                )
            if not self.OAUTH_ENCRYPTION_KEYS:
                raise ValueError(
                    "DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS must be configured."
                )
        elif _is_unconfigured(self.OPENAI_GATEWAY_API_KEY):
            raise ValueError(
                "OPENAI_GATEWAY_API_KEY must be configured for mock login."
            )
        return self


settings = Settings()


class ChainlitSettings(BaseSettings):
    """Validate unprefixed Chainlit persistence and OIDC settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        extra="ignore",
    )

    DATABASE_URL: PostgresDsn = Field(repr=False)
    CHAINLIT_AUTH_SECRET: str = Field(repr=False)
    BUCKET_NAME: str
    APP_AWS_ACCESS_KEY: str = Field(repr=False)
    APP_AWS_SECRET_KEY: str = Field(repr=False)
    APP_AWS_REGION: str
    DEV_AWS_ENDPOINT: HttpUrlStr
    OAUTH_GENERIC_CLIENT_ID: str | None = None
    OAUTH_GENERIC_CLIENT_SECRET: str | None = Field(default=None, repr=False)
    OAUTH_GENERIC_SCOPES: str | None = None
    OAUTH_GENERIC_NAME: str = Field(default="generic", pattern=r"^[A-Za-z0-9_-]+$")
    CHAINLIT_URL: HttpUrlStr | None = None

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
        """Require OIDC client settings only when OAuth is selected."""
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
        assert self.CHAINLIT_URL is not None
        if not self.CHAINLIT_URL.startswith("https://"):
            raise ValueError("CHAINLIT_URL must use HTTPS for OAuth.")
        if "openid" not in (self.OAUTH_GENERIC_SCOPES or "").split():
            raise ValueError("OAUTH_GENERIC_SCOPES must include openid.")
        return self


@cache
def get_chainlit_settings() -> ChainlitSettings:
    """Load and validate the unprefixed settings once per process."""
    return ChainlitSettings()
