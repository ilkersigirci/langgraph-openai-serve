"""One-shot checkpoint schema initialization command."""

import asyncio
import logging

from lgos_demo_api.checkpointer import setup_postgres_schema
from lgos_demo_api.loggers.setup import setup_logging
from lgos_demo_api.settings import settings

logger = logging.getLogger(__name__)


async def setup_checkpoint_schema() -> None:
    """Initialize the configured PostgreSQL checkpoint schema."""
    logger.info("Initializing PostgreSQL checkpoint schema")
    await setup_postgres_schema(settings.POSTGRES_URI)
    logger.info("PostgreSQL checkpoint schema is ready")


def main() -> None:
    """Run checkpoint schema initialization as a deployment task."""
    setup_logging()
    asyncio.run(setup_checkpoint_schema())


if __name__ == "__main__":
    main()
