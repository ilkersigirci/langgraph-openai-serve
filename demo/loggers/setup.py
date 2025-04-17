"""Logger initialization methods."""

import logging
import logging.config

from demo.loggers.configs.default import DEFAULT_LOGGER_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config: dict | None = None,
    default_level: int = logging.INFO,
) -> None:
    """Set up the logger using default or custom configuration.

    Args:
        logging_config: Custom logging configuration.
        default_level: Default level of the logger.
    """
    loaded_config = DEFAULT_LOGGER_CONFIG if logging_config is None else logging_config

    try:
        logging.config.dictConfig(loaded_config)
    except Exception as e:
        message = f"Error when loading given logging configuration. Using default configs. Error: {e}"
        print(message)
        logging.basicConfig(level=default_level)


if __name__ == "__main__":
    setup_logging(default_level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    try:
        print(1 / 0)
    except Exception:
        logger.exception("unable print!")
