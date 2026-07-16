"""One-shot Chainlit persistence schema migration command."""

import asyncio
import logging

from demo.ui.chainlit_ui.database import setup_chainlit_schema
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class DatabaseSettings(BaseSettings):
    """Native Chainlit database configuration used by the migration command."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str


async def setup_database() -> None:
    """Apply pending Chainlit schema migrations."""
    logger.info("Initializing PostgreSQL Chainlit persistence schema")
    await setup_chainlit_schema(DatabaseSettings().DATABASE_URL)
    logger.info("PostgreSQL Chainlit persistence schema is ready")


def main() -> None:
    """Run the schema migration command."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(setup_database())


if __name__ == "__main__":
    main()
