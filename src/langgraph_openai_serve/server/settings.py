"""Settings for ``lgos serve`` and ``lgos worker``."""

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server settings read from explicit values and the process environment."""

    model_config = SettingsConfigDict(
        env_prefix="LGOS_",
        # `.env` files often leave unused values empty; treat them as unset.
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    HOST: str = "127.0.0.1"
    PORT: Annotated[int, Field(ge=1, le=65535)] = 8000
    CORS_ORIGINS: list[str] = []
    # Without a database, checkpoints and run leases live in this process only.
    POSTGRES_URI: SecretStr | None = None
    # One connection serves checkpoints; each running interrupt holds another.
    POSTGRES_POOL_SIZE: Annotated[int, Field(ge=2)] = 5
    INTERRUPT_TTL_MINUTES: PositiveInt = 43200
    # 0 disables the sweep, for example when one scheduled job runs it instead.
    INTERRUPT_SWEEP_INTERVAL_MINUTES: NonNegativeInt = 5
    BACKGROUND: Literal["none", "memory", "hatchet"] = "none"
    HATCHET_WORKER_SLOTS: PositiveInt = 4
