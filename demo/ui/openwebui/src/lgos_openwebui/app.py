"""Run the pinned Open WebUI application with the LGOS integration."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from importlib import import_module
from typing import Any

from starlette.types import ASGIApp

from .connection import without_lgos_models
from .install import install_functions

logger = logging.getLogger(__name__)


def create_app() -> ASGIApp:
    """Load Open WebUI with the Functions installed and owning the lgos/ models."""
    # Open WebUI is installed only in its image, not in this project.
    open_webui: Any = import_module("open_webui.main").app
    models: Any = import_module("open_webui.utils.models")
    # get_all_base_models looks this up on every call, so the filter also reaches
    # the admin model list, whose route imported get_all_base_models by name.
    models.fetch_openai_models = without_lgos_models(models.fetch_openai_models)
    open_webui_lifespan = open_webui.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: ASGIApp) -> AsyncGenerator[object, None]:
        async with open_webui_lifespan(app) as state:
            # Open WebUI's startup creates the bootstrap admin in an empty database.
            token = await _admin_token()
            for function_id, action in (await install_functions(app, token)).items():
                logger.info(
                    "%s Open WebUI Function %s", action.capitalize(), function_id
                )
            yield state

    open_webui.router.lifespan_context = lifespan
    return open_webui


async def _admin_token() -> str:
    """
    Return a short-lived token for an Open WebUI admin.

    Signing in would need password sign-in, which the SSO overlay disables.
    The claims match the ones Open WebUI's own automations act with in-process.
    """
    users: Any = import_module("open_webui.models.users").Users
    auths: Any = import_module("open_webui.models.auths").Auths
    create_token: Any = import_module("open_webui.utils.auth").create_token
    admin = await users.get_super_admin_user()
    if admin is None:
        msg = (
            "Open WebUI has no admin to install the Functions as; set "
            "WEBUI_ADMIN_EMAIL and WEBUI_ADMIN_PASSWORD for an empty database."
        )
        raise RuntimeError(msg)
    auth = await auths.get_auth_by_id(admin.id)
    return create_token(
        {"id": admin.id, "typ": "automation", "session_stamp": auth.session_stamp},
        expires_delta=timedelta(minutes=5),
    )
