"""The ``lgos`` command: serve a graph registry or run its Hatchet worker."""

import logging
import os
import pkgutil
import sys
from typing import TYPE_CHECKING, Any

import click
import uvicorn

if TYPE_CHECKING:
    from fastapi import FastAPI

# Like UVICORN_APP, the environment can name the target once per image.
_REGISTRY = click.Argument(["registry"], envvar="LGOS_REGISTRY")


@click.group()
@click.version_option(package_name="langgraph_openai_serve")
def main() -> None:
    """Serve a graph registry or run its Hatchet worker."""


# `lgos serve` is Uvicorn's own command with the registry in place of the app, so
# every Uvicorn option, and the UVICORN_* variable for it, applies unchanged.
@main.command(
    params=[
        _REGISTRY,
        *(
            param
            for param in uvicorn.main.params
            if param.name not in {"app", "factory"}
        ),
    ],
    context_settings={
        "auto_envvar_prefix": "UVICORN",
        # Flags and UVICORN_* variables override these LGOS defaults.
        "default_map": {"access_log": False},
    },
)
def serve(registry: str, log_config: str | None, **options: Any) -> None:
    """Serve REGISTRY ($LGOS_REGISTRY), a module:attribute factory, with Uvicorn."""
    try:
        from langgraph_openai_serve.server.logging import logging_config
        from langgraph_openai_serve.server.settings import ServerSettings
    except ModuleNotFoundError as exc:
        raise _missing_server_extra(exc) from exc
    # Uvicorn's worker count: the option, else WEB_CONCURRENCY, else one.
    workers = options["workers"] or int(os.getenv("WEB_CONCURRENCY", "1"))
    # Uvicorn loads --env-file only once it starts, so read it here as well.
    settings = ServerSettings(_env_file=options["env_file"])
    if workers > 1 and settings.BACKGROUND == "memory":
        msg = (
            "LGOS_BACKGROUND=memory keeps background Responses in one process; "
            "run one worker or set LGOS_BACKGROUND=hatchet."
        )
        raise click.UsageError(msg)
    # Uvicorn imports the app again in its worker and reload processes, so the
    # registry travels through the environment they inherit.
    os.environ["LGOS_REGISTRY"] = registry
    click.get_current_context().invoke(
        uvicorn.main,
        app=f"{__name__}:{_create_app.__name__}",
        factory=True,
        log_config=log_config
        or logging_config(_application(registry), root_level=logging.WARNING),
        **options,
    )


@main.command(params=[_REGISTRY])
def worker(registry: str) -> None:
    """Run background Responses for REGISTRY ($LGOS_REGISTRY) on Hatchet."""
    try:
        from langgraph_openai_serve.server.hatchet import run_worker
        from langgraph_openai_serve.server.logging import configure_logging
        from langgraph_openai_serve.server.settings import ServerSettings
    except ModuleNotFoundError as exc:
        raise _missing_server_extra(exc) from exc
    # Like Uvicorn, resolve modules from the working directory first.
    sys.path.insert(0, "")
    factory = pkgutil.resolve_name(registry)
    # Hatchet forwards task logs at the root logger's level.
    configure_logging(_application(registry), root_level=logging.INFO)
    run_worker(factory, ServerSettings())


def _create_app() -> "FastAPI":
    """Build the application in each process Uvicorn starts for ``lgos serve``."""
    from langgraph_openai_serve.server.app import create_app

    return create_app(pkgutil.resolve_name(os.environ["LGOS_REGISTRY"]))


def _application(registry: str) -> str:
    """Name the registry's top-level package, which logs at INFO with LGOS."""
    return registry.partition(":")[0].split(".")[0]


def _missing_server_extra(exc: ModuleNotFoundError) -> click.ClickException:
    return click.ClickException(f"{exc}; install langgraph-openai-serve[server]")
