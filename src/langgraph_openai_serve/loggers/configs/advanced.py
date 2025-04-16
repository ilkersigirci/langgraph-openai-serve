"""Advanced logging configuration."""

ADVANCED_LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "error": {
            "format": "%(asctime)s - %(name)s - %(levelname)s <PID %(process)d:%(processName)s> %(name)s.%(funcName)s(): %(message)s"
        },
    },
    "handlers": {
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "logs/info.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8",
        },
        "warn_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "WARN",
            "formatter": "standard",
            "filename": "logs/warn.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8",
        },
        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "error",
            "filename": "logs/errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8",
        },
        "critical_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "CRITICAL",
            "formatter": "standard",
            "filename": "logs/critical.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8",
        },
        "debug_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "logs/debug.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8",
        },
        "root_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "logs/logs.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8",
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "error",
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "DEBUG",
            "handlers": ["console", "error_console", "root_file_handler"],
            "propagate": True,
        },
        "main": {
            "level": "DEBUG",
            "handlers": [
                "info_file_handler",
                "warn_file_handler",
                "error_file_handler",
                "critical_file_handler",
                "debug_file_handler",
            ],
            "propagate": False,
        },
        "werkzeug": {
            "level": "DEBUG",
            "handlers": [
                "info_file_handler",
                "warn_file_handler",
                "error_file_handler",
                "critical_file_handler",
                "debug_file_handler",
            ],
            "propagate": True,
        },
        "api.app_server": {
            "level": "DEBUG",
            "handlers": [
                "info_file_handler",
                "warn_file_handler",
                "error_file_handler",
                "critical_file_handler",
                "debug_file_handler",
            ],
            "propagate": True,
        },
    },
}
