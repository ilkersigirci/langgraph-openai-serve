"""Default logging configuration."""

DEFAULT_LOGGER_CONFIG = {
    "version": 1,
    # dictConfig otherwise disables library loggers initialized before the demo
    # configures logging.
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s :: %(name)s :: %(message)s"},
        "extended": {
            "format": "%(asctime)-20s :: %(levelname)-8s :: [%(process)d]%(processName)s :: %(threadName)s[%(thread)d] :: %(pathname)s:%(lineno)d - %(funcName)s :: %(message)s"
        },
        "aligned": {
            "format": "{asctime} :: {levelname:<8s}:: {pathname:<10s}:{lineno} :: {message}",
            "style": "{",
        },
        "colored": {
            "()": "lgos_demo_api.loggers.formatters.ColoredFormatter",
            "format": "%(asctime)-20s :: %(name)-8s :: %(levelname)-8s :: %(pathname)s:%(lineno)d :: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "info": {
            "()": "lgos_demo_api.loggers.filters.InfoFilter",
        },
        "cwd": {
            "()": "lgos_demo_api.loggers.filters.CwdFilter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
            "formatter": "colored",
            "filters": ["cwd"],
        },
    },
    "loggers": {
        "": {  # An empty logger name configures the root logger.
            "level": "WARNING",
            "propagate": True,
        },
        "__main__": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": True,
        },
        "langgraph_openai_serve": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": True,
        },
    },
}
