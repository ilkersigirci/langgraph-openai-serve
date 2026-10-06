"""The ``lgos`` command: serve a graph registry or run its Hatchet worker."""

import argparse
import logging
import os
import pkgutil
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``lgos serve REGISTRY`` or ``lgos worker REGISTRY``."""
    parser = argparse.ArgumentParser(prog="lgos", description=__doc__)
    parser.add_argument("command", choices=["serve", "worker"])
    # Like Uvicorn's UVICORN_APP, the environment can name the target once per image.
    parser.add_argument(
        "registry",
        nargs="?",
        default=os.getenv("LGOS_REGISTRY"),
        help="registry factory as module:attribute (default: $LGOS_REGISTRY)",
    )
    arguments = parser.parse_args(argv)
    if arguments.registry is None:
        parser.error("name a registry factory or set LGOS_REGISTRY")
    try:
        import uvicorn

        from langgraph_openai_serve.server.app import create_app
        from langgraph_openai_serve.server.hatchet import run_worker
        from langgraph_openai_serve.server.logging import (
            configure_logging,
            logging_config,
        )
        from langgraph_openai_serve.server.settings import ServerSettings
    except ModuleNotFoundError as exc:
        parser.exit(1, f"lgos: {exc}; install langgraph-openai-serve[server]\n")

    # Like Uvicorn, resolve modules from the working directory first.
    sys.path.insert(0, "")
    factory = pkgutil.resolve_name(arguments.registry)
    # The registry's top-level package logs at INFO with LGOS and Uvicorn.
    application = arguments.registry.partition(":")[0].split(".")[0]
    settings = ServerSettings()
    if arguments.command == "worker":
        # Hatchet forwards task logs at the root logger's level.
        configure_logging(application, root_level=logging.INFO)
        run_worker(factory, settings)
        return
    uvicorn.run(
        create_app(factory, settings=settings),
        host=settings.HOST,
        port=settings.PORT,
        access_log=False,
        log_config=logging_config(application, root_level=logging.WARNING),
    )
