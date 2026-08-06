"""Selectable demo login callbacks."""

import chainlit as cl

from lgos_chainlit.settings import settings

MOCK_USER_IDENTIFIER = "demo-user"


async def mock_login(_username: str, _password: str) -> cl.User:
    """Return the shared user used by the local demo login form."""
    return cl.User(
        identifier=MOCK_USER_IDENTIFIER,
        display_name="Demo User",
        metadata={"provider": "mock"},
    )


async def oauth_login(
    _provider_id: str,
    _token: str,
    _raw_user_data: dict[str, str],
    default_user: cl.User,
    _id_token: str | None = None,
) -> cl.User:
    """Accept the user returned by Chainlit's configured OAuth provider."""
    return default_user


def register_auth_callback() -> None:
    """Register the callback selected by the demo configuration."""
    if settings.LOGIN_TYPE == "mock":
        cl.password_auth_callback(mock_login)
    else:
        cl.oauth_callback(oauth_login)
