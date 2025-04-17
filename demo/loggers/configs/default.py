"""Default logging configuration."""

DEFAULT_LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # NOTE: Very important
    "formatters": {
        "simple": {"format": "%(asctime)s :: %(name)s :: %(message)s"},
        "extended": {
            "format": "%(asctime)-20s :: %(levelname)-8s :: [%(process)d]%(processName)s :: %(threadName)s[%(thread)d] :: %(pathname)s:%(lineno)d - %(funcName)s :: %(message)s"
        },
        "aligned": {
            "format": "{asctime} :: {levelname:<8s}:: {pathname:<10s}:{lineno} :: {message}",
            "style": "{",
        },
        "base": {
            "format": "%(asctime)-20s :: %(levelname)-8s :: %(pathname)s:%(lineno)d:: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "colored": {
            "()": "demo.loggers.formatters.ColoredFormatter",
            "format": "%(asctime)-20s :: %(name)-8s :: %(levelname)-8s :: %(pathname)s:%(lineno)d :: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "info": {
            "()": "demo.loggers.filters.InfoFilter",
        },
        "cwd": {
            "()": "demo.loggers.filters.CwdFilter",
        },
        # "path_shortener": {
        #     "()": "demo.loggers.filters.PathShortenerFilter",
        # }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
            "formatter": "colored",
            "filters": ["cwd"],
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "filename": "demo.log",
            "formatter": "base",
            "filters": ["cwd"],
        },
    },
    "loggers": {
        # NOTE: Default root logger parameters. It is not recommended to
        # modify root logger.
        "": {  # root logger
            # "level": "NOTSET",  # logs everything
            "level": "WARNING",
            # "handlers": ["console"],
            "propagate": True,
        },
        "__main__": {  # if __name__ == '__main__'
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": True,  # Inherit root handlers
        },
        "langgraph_openai_serve": {
            "level": "DEBUG",
            "handlers": ["console"],  # ,file_handler
            "propagate": True,  # Inherit root handlers
        },
    },
}
