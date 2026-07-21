"""Selectable demo login callbacks and authenticated-user helpers."""

from __future__ import annotations

import chainlit as cl

from lgos_chainlit.settings import settings

MOCK_USER_IDENTIFIER = "demo-user"


if settings.LOGIN_TYPE == "mock":

    @cl.password_auth_callback
    async def mock_login(_username: str, _password: str) -> cl.User:
        """Return the shared user used by the local demo login form."""
        return cl.User(
            identifier=MOCK_USER_IDENTIFIER,
            display_name="Demo User",
            metadata={"provider": "mock"},
        )

else:

    @cl.oauth_callback
    async def oauth_callback(
        _provider_id: str,
        _token: str,
        _raw_user_data: dict[str, str],
        default_user: cl.User,
        _id_token: str | None = None,
    ) -> cl.User:
        """Accept the user returned by Chainlit's configured OAuth provider."""
        return default_user


def authenticated_user_identifier() -> str:
    """Return the unique identifier of the current authenticated user."""
    user = cl.user_session.get("user")
    identifier = getattr(user, "identifier", None)
    if not isinstance(identifier, str) or not identifier:
        raise RuntimeError("Chainlit request has no authenticated user.")
    return identifier
