"""Chainlit login integration and request-scoped gateway credentials."""

import logging
import os
from contextvars import ContextVar
from time import time
from typing import cast
from uuid import uuid4

import chainlit as cl
import httpx2
from authlib.common.errors import AuthlibBaseError
from authlib.oidc.core import UserInfo
from chainlit.auth import clear_auth_cookie, create_jwt
from chainlit.auth.cookie import OAuth2PasswordBearerWithCookie
from chainlit.auth.jwt import decode_jwt
from chainlit.config import config
from chainlit.context import ChainlitContextException
from chainlit.data import get_data_layer
from chainlit.oauth_providers import OAuthProvider, providers
from chainlit.server import logout as chainlit_logout
from chainlit.session import WebsocketSession
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from joserfc.errors import JoseError
from jwt import PyJWTError
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from lgos_chainlit.auth.oauth_client import oidc_client, oidc_metadata, token_client
from lgos_chainlit.auth.oauth_tokens import (
    OAuthLoginRequired,
    OAuthTokens,
    access_token,
    delete_oauth_session,
    save_oauth_tokens,
)
from lgos_chainlit.settings import get_chainlit_settings, settings

logger = logging.getLogger(__name__)
MOCK_USER_IDENTIFIER = "demo-user"
SESSION_CLAIM = "oauth_session"
_request_token: ContextVar[str | None] = ContextVar("oauth_request_token", default=None)
_browser_auth = OAuth2PasswordBearerWithCookie(tokenUrl="/login", auto_error=False)
router = APIRouter()


def session_identity(token: str | None) -> tuple[str, str]:
    """Read session identity only from Chainlit's verified JWT, never persisted metadata."""
    if not token:
        raise OAuthLoginRequired()
    try:
        user = decode_jwt(token)
    except (PyJWTError, ValueError, TypeError):
        raise OAuthLoginRequired() from None
    session_id = user.metadata.get(SESSION_CLAIM)
    if not isinstance(session_id, str) or not session_id:
        raise OAuthLoginRequired()
    return session_id, user.identifier


class GatewayRequestContextMiddleware:
    """Expose HTTP credentials to gateway calls; Chainlit owns UI authentication."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        context_token = _request_token.set(await _browser_auth(Request(scope)))
        try:
            await self.app(scope, receive, send)
        finally:
            _request_token.reset(context_token)


async def gateway_credential() -> str:
    if not settings.ENABLE_OAUTH_TOKEN_FORWARDING:
        assert settings.GATEWAY_API_KEY is not None
        return settings.GATEWAY_API_KEY
    # HTTP discovery has no Chainlit context. Chat callbacks use the native
    # socket session, even when Socket.IO's transport task inherited an HTTP token.
    token = _request_token.get()
    try:
        session = cl.context.session
    except ChainlitContextException:
        session = None
    if isinstance(session, WebsocketSession):
        token = session.token
    session_id, identifier = session_identity(token)
    if isinstance(session, WebsocketSession) and (
        session.user is None or session.user.identifier != identifier
    ):
        raise OAuthLoginRequired()
    return await access_token(session_id, identifier)


async def mock_login(_username: str, _password: str) -> cl.User:
    return cl.User(
        identifier=MOCK_USER_IDENTIFIER,
        display_name="Demo User",
        metadata={"provider": "mock"},
    )


class OIDCLoginButton(OAuthProvider):
    """Advertise the native login button; the parent app owns both OAuth routes."""

    def __init__(self) -> None:
        self.env = [
            "OAUTH_GENERIC_CLIENT_ID",
            "OAUTH_GENERIC_CLIENT_SECRET",
            "DEMO_CHAINLIT_OAUTH_ISSUER",
        ]
        self.id = get_chainlit_settings().OAUTH_GENERIC_NAME


async def oauth_login(*_args: object) -> None:
    # Fail closed if a deployment accidentally exposes Chainlit's non-PKCE callback.
    raise OAuthLoginRequired()


def _check_provider(provider_id: str) -> None:
    if provider_id != get_chainlit_settings().OAUTH_GENERIC_NAME:
        raise HTTPException(404, "Unknown OAuth provider.")


@router.get("/auth/oauth/{provider_id}")
async def oauth_authorize(provider_id: str, request: Request):
    _check_provider(provider_id)
    await oidc_metadata()
    native = get_chainlit_settings()
    return await oidc_client().authorize_redirect(
        request,
        f"{native.CHAINLIT_URL}/auth/oauth/{provider_id}/callback",
        resource=settings.OAUTH_RESOURCE,
        prompt=OIDCLoginButton().get_prompt(),
    )


@router.get("/auth/oauth/{provider_id}/callback")
async def oauth_callback(provider_id: str, request: Request):
    _check_provider(provider_id)
    try:
        await oidc_metadata()
        result = await oidc_client().authorize_access_token(
            request,
            resource=settings.OAUTH_RESOURCE,
            leeway=10,
        )
        tokens = (
            OAuthTokens.model_validate(result)
            if settings.ENABLE_OAUTH_TOKEN_FORWARDING
            else None
        )
        # Only Authlib's parsed ID token, never raw userinfo from a token response.
        identity = result.get("userinfo")
        if not isinstance(identity, UserInfo):
            raise ValueError("Missing validated ID token.")
        subject = identity.get("sub")
        if not isinstance(subject, str) or not subject:
            raise ValueError("Missing verified subject.")
    except (httpx2.HTTPError, AuthlibBaseError, JoseError, ValueError):
        request.session.clear()
        return RedirectResponse(
            f"{get_chainlit_settings().CHAINLIT_URL}/login?error=oauth_callback_error",
            status_code=302,
        )
    user = cl.User(identifier=subject, metadata={"provider": provider_id})
    layer = get_data_layer()
    assert layer is not None
    await layer.create_user(user)
    if tokens is not None:
        session_id = uuid4().hex
        await save_oauth_tokens(
            session_id, subject, tokens, time() + config.project.user_session_timeout
        )
        try:
            previous_identity = session_identity(await _browser_auth(request))
        except OAuthLoginRequired:
            pass
        else:
            await delete_oauth_session(*previous_identity)
        # Add the session after persistence so it stays browser-local.
        user.metadata[SESSION_CLAIM] = session_id
    response = RedirectResponse(
        f"{get_chainlit_settings().CHAINLIT_URL}/login/callback?success=true",
        status_code=302,
    )
    clear_auth_cookie(request, response)
    # Chainlit 2.11.1's cookie helper sets Secure only with SameSite=None.
    response.set_cookie(
        os.environ.get("CHAINLIT_AUTH_COOKIE_NAME", "access_token"),
        create_jwt(user),
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=config.project.user_session_timeout,
    )
    request.session.clear()
    return response


@router.post("/logout")
async def oauth_logout(request: Request, response: Response):
    tokens: OAuthTokens | None = None
    if settings.ENABLE_OAUTH_TOKEN_FORWARDING:
        try:
            identity = session_identity(await _browser_auth(request))
        except OAuthLoginRequired:
            pass
        else:
            tokens = await delete_oauth_session(*identity)
    request.session.clear()
    if tokens is not None:
        try:
            metadata = await oidc_metadata()
            if endpoint := metadata.get("revocation_endpoint"):
                async with token_client() as client:
                    revoked = await client.revoke_token(
                        endpoint,
                        token=tokens.refresh_token or tokens.access_token,
                        token_type_hint="refresh_token"
                        if tokens.refresh_token
                        else "access_token",
                    )
                    revoked.raise_for_status()
        except (httpx2.HTTPError, AuthlibBaseError, ValueError):
            # Never log protocol responses or restore a deleted gateway grant.
            logger.warning(
                "OAuth provider revocation failed; local gateway grant was removed."
            )
    return await chainlit_logout(request, response)


def configure_auth(app: FastAPI) -> None:
    """Configure the selected Chainlit login mode before mounting the UI."""
    if settings.LOGIN_TYPE == "mock":
        cl.password_auth_callback(mock_login)
        return

    config.code.password_auth_callback = None
    config.code.header_auth_callback = None
    # Chainlit infers a narrower registry type from its built-in providers.
    cast(list[OAuthProvider], providers)[:] = [OIDCLoginButton()]
    cl.oauth_callback(oauth_login)
    app.include_router(router)
    if settings.ENABLE_OAUTH_TOKEN_FORWARDING:
        app.add_middleware(GatewayRequestContextMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=get_chainlit_settings().CHAINLIT_AUTH_SECRET,
        session_cookie="lgos_oauth_state",
        max_age=300,
        https_only=True,
        same_site="lax",
    )
