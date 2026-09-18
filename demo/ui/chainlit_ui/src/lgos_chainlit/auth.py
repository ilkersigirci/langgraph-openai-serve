"""Configure demo login policy and gateway credentials."""

from functools import cache

import chainlit as cl
from chainlit_utils.sso.chainlit import ChainlitOAuth, delegated_oauth_credential
from chainlit_utils.sso.oidc import OidcClient, OidcConfig
from chainlit_utils.sso.tokens import OAuthTokenStore
from fastapi import FastAPI

from lgos_chainlit.settings import get_chainlit_settings, settings

SESSION_CLAIM = "oauth_session"


@cache
def oidc_client() -> OidcClient:
    """Build the OIDC client from the demo's validated settings."""
    native = get_chainlit_settings()
    assert settings.OAUTH_ISSUER is not None
    assert native.OAUTH_GENERIC_CLIENT_ID is not None
    assert native.OAUTH_GENERIC_CLIENT_SECRET is not None
    assert native.OAUTH_GENERIC_SCOPES is not None
    return OidcClient(
        OidcConfig(
            issuer=settings.OAUTH_ISSUER,
            client_id=native.OAUTH_GENERIC_CLIENT_ID,
            client_secret=native.OAUTH_GENERIC_CLIENT_SECRET,
            scopes=native.OAUTH_GENERIC_SCOPES,
            resource=settings.OAUTH_RESOURCE,
            client_auth_method=settings.OAUTH_CLIENT_AUTH_METHOD,
        )
    )


@cache
def token_store() -> OAuthTokenStore:
    """Use the demo's encryption keys and existing PostgreSQL grant table."""
    return OAuthTokenStore(
        oidc_client(),
        lambda: [key.get_secret_value() for key in settings.OAUTH_ENCRYPTION_KEYS],
        table_name="lgos_chainlit_oauth_sessions",
    )


async def gateway_credential() -> str:
    """Return the shared key or the current user's delegated access token."""
    if not settings.ENABLE_OAUTH_TOKEN_FORWARDING:
        assert settings.OPENAI_GATEWAY_API_KEY is not None
        return settings.OPENAI_GATEWAY_API_KEY
    return await delegated_oauth_credential(
        token_store().access_token,
        session_claim=SESSION_CLAIM,
    )


async def mock_login(_username: str, _password: str) -> cl.User:
    return cl.User(
        identifier="demo-user",
        display_name="Demo User",
        metadata={"provider": "mock"},
    )


def configure_auth(app: FastAPI) -> None:
    """Configure the selected login mode before mounting Chainlit."""
    if settings.LOGIN_TYPE == "mock":
        cl.password_auth_callback(mock_login)
        return

    native = get_chainlit_settings()
    assert native.CHAINLIT_URL is not None
    ChainlitOAuth(
        provider_id=native.OAUTH_GENERIC_NAME,
        chainlit_url=native.CHAINLIT_URL,
        auth_secret=native.CHAINLIT_AUTH_SECRET,
        oidc=oidc_client(),
        provider_env=(
            "OAUTH_GENERIC_CLIENT_ID",
            "OAUTH_GENERIC_CLIENT_SECRET",
            "DEMO_CHAINLIT_OAUTH_ISSUER",
        ),
        token_store=token_store() if settings.ENABLE_OAUTH_TOKEN_FORWARDING else None,
        session_claim=SESSION_CLAIM,
        state_cookie="lgos_oauth_state",
    ).configure(app)
