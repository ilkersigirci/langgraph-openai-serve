"""Authlib owns discovery, PKCE, OIDC validation, and OAuth protocol requests."""

from functools import cache
from typing import Any

from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuth, StarletteOAuth2App

from lgos_chainlit.settings import get_chainlit_settings, settings


def _client_kwargs() -> dict[str, Any]:
    return {
        "token_endpoint_auth_method": settings.OAUTH_CLIENT_AUTH_METHOD,
        "revocation_endpoint_auth_method": settings.OAUTH_CLIENT_AUTH_METHOD,
        "timeout": 10,
        "trust_env": False,
    }


@cache
def oidc_client() -> StarletteOAuth2App:
    native = get_chainlit_settings()
    assert settings.OAUTH_ISSUER is not None
    client = OAuth().register(
        "oidc",
        client_id=native.OAUTH_GENERIC_CLIENT_ID,
        client_secret=native.OAUTH_GENERIC_CLIENT_SECRET,
        server_metadata_url=f"{settings.OAUTH_ISSUER.rstrip('/')}/.well-known/openid-configuration",
        client_kwargs={
            **_client_kwargs(),
            "scope": native.OAUTH_GENERIC_SCOPES,
            "code_challenge_method": "S256",
        },
    )
    assert isinstance(client, StarletteOAuth2App)
    return client


async def oidc_metadata() -> dict[str, Any]:
    metadata = await oidc_client().load_server_metadata()
    if metadata.get("issuer") != settings.OAUTH_ISSUER:
        raise ValueError("OIDC discovery issuer does not match the configured issuer.")
    if "S256" not in metadata.get("code_challenge_methods_supported", []):
        raise ValueError("OIDC provider must advertise S256 PKCE support.")
    if settings.OAUTH_CLIENT_AUTH_METHOD not in metadata.get(
        "token_endpoint_auth_methods_supported", ["client_secret_basic"]
    ):
        raise ValueError(
            "OIDC provider does not support the configured client authentication method."
        )
    return metadata


def token_client() -> AsyncOAuth2Client:
    native = get_chainlit_settings()
    return AsyncOAuth2Client(
        client_id=native.OAUTH_GENERIC_CLIENT_ID,
        client_secret=native.OAUTH_GENERIC_CLIENT_SECRET,
        **_client_kwargs(),
    )
