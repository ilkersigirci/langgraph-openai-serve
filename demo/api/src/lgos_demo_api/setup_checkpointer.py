"""One-shot LangGraph persistence schema initialization command."""

import asyncio
import logging

from lgos_demo_api.checkpointer import setup_postgres_schema
from lgos_demo_api.logging import configure_logging
from lgos_demo_api.settings import settings

logger = logging.getLogger(__name__)


async def setup_persistence_schema() -> None:
    """Initialize the configured PostgreSQL persistence schemas."""
    logger.info("demo.persistence_schema.initializing")
    await setup_postgres_schema(settings.POSTGRES_URI)
    logger.info("demo.persistence_schema.ready")


def main() -> None:
    """Run persistence schema initialization as a deployment task."""
    configure_logging()
    asyncio.run(setup_persistence_schema())


if __name__ == "__main__":
    main()
