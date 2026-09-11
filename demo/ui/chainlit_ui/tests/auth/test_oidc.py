"""Exercise real Authlib/Chainlit HTTP flows against a signing OIDC provider."""

from base64 import b64decode, urlsafe_b64encode
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import chainlit as cl
import httpx
import httpx2
import jwt
import pytest
from authlib.integrations.httpx_client import AsyncOAuth2Client
from chainlit.utils import mount_chainlit
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request, Response
from jwt.algorithms import RSAAlgorithm

from lgos_chainlit.auth import chainlit as auth
from lgos_chainlit.auth import oauth_client
from lgos_chainlit.auth.oauth_tokens import OAuthLoginRequired, OAuthTokens
from lgos_chainlit.settings import get_chainlit_settings, settings
from lgos_chainlit.utils import clients


@dataclass
class OIDCProvider:
    auth_method: str = "client_secret_basic"
    resource: str | None = "https://llm.example/"
    key: rsa.RSAPrivateKey = field(
        default_factory=lambda: rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
    )
    codes: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    claims_override: dict = field(default_factory=dict)
    token_override: dict = field(default_factory=dict)
    exchanges: int = 0
    revocations: list[str] = field(default_factory=list)
    revocation_status: int = 200
    invalid_signature: bool = False

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("openid-configuration"):
            return httpx2.Response(
                200,
                json={
                    "issuer": "https://id.example",
                    "authorization_endpoint": "https://id.example/authorize",
                    "token_endpoint": "https://id.example/token",
                    "jwks_uri": "https://id.example/jwks",
                    "revocation_endpoint": "https://id.example/revoke",
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": [self.auth_method],
                    "id_token_signing_alg_values_supported": ["RS256"],
                },
            )
        if request.url.path == "/jwks":
            key = RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
            return httpx2.Response(
                200, json={"keys": [{**key, "kid": "test-key", "use": "sig"}]}
            )
        form = parse_qs(request.content.decode(), keep_blank_values=True)
        if self.auth_method == "client_secret_basic":
            scheme, credentials = request.headers["authorization"].split()
            assert scheme == "Basic"
            assert b64decode(credentials).decode() == "chainlit-client:client-secret"
            assert "client_secret" not in form
        else:
            assert form["client_id"] == ["chainlit-client"]
            assert form["client_secret"] == ["client-secret"]
            assert "authorization" not in request.headers
        if request.url.path == "/revoke":
            self.revocations.append(form["token"][0])
            return httpx2.Response(self.revocation_status)
        assert request.url.path == "/token"
        self.exchanges += 1
        code = form["code"][0]
        params = self.codes.pop(code)
        assert form.get("resource") == ([self.resource] if self.resource else None)
        assert form["redirect_uri"] == [
            "https://chat.example/auth/oauth/generic/callback"
        ]
        challenge = (
            urlsafe_b64encode(sha256(form["code_verifier"][0].encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert params["code_challenge"] == [challenge]
        claims = {
            "iss": "https://id.example",
            "sub": "alice",
            "aud": "chainlit-client",
            "iat": int(time()),
            "exp": int(time()) + 3600,
            "nonce": params["nonce"][0],
            **self.claims_override,
        }
        return httpx2.Response(
            200,
            json={
                "access_token": f"access-{code}",
                "refresh_token": f"refresh-{code}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "id_token": jwt.encode(
                    claims,
                    rsa.generate_private_key(public_exponent=65537, key_size=2048)
                    if self.invalid_signature
                    else self.key,
                    algorithm="RS256",
                    headers={"kid": "test-key"},
                ),
                **self.token_override,
            },
        )


@dataclass
class OAuthApp:
    app: FastAPI
    provider: OIDCProvider
    sessions: dict[tuple[str, str], OAuthTokens]
    persisted: dict[str, cl.User]
    gateway_authorizations: list[str]

    def browser(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="https://chat.example"
        )

    async def start(
        self, browser: httpx.AsyncClient, code: str
    ) -> dict[str, list[str]]:
        response = await browser.get("/auth/oauth/generic")
        assert response.status_code == 302
        params = parse_qs(
            urlparse(response.headers["location"]).query, keep_blank_values=True
        )
        assert params["code_challenge_method"] == ["S256"]
        assert params.get("resource") == (
            [self.provider.resource] if self.provider.resource else None
        )
        self.provider.codes[code] = params
        return params

    async def login(self, browser: httpx.AsyncClient, code: str) -> httpx.Response:
        params = await self.start(browser, code)
        return await browser.get(
            "/auth/oauth/generic/callback",
            params={"state": params["state"][0], "code": code},
        )


@pytest.fixture
async def oauth_app(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> AsyncIterator[OAuthApp]:
    import chainlit.data
    from chainlit.server import app as chainlit_app

    # mount_chainlit modifies a singleton; keep each ASGI application's middleware isolated.
    monkeypatch.setattr(chainlit_app, "middleware_stack", None)
    monkeypatch.setattr(
        chainlit_app, "user_middleware", chainlit_app.user_middleware.copy()
    )

    for name, value in {
        "CHAINLIT_AUTH_SECRET": "test-signing-secret-with-at-least-32-bytes",
        "DATABASE_URL": "postgresql://user:password@unused/db",
        "CHAINLIT_URL": "https://chat.example",
        "OAUTH_GENERIC_CLIENT_ID": "chainlit-client",
        "OAUTH_GENERIC_CLIENT_SECRET": "client-secret",
        "OAUTH_GENERIC_SCOPES": "openid offline_access llm:invoke",
        "OAUTH_GENERIC_NAME": "generic",
        "DEMO_CHAINLIT_OAUTH_ISSUER": "https://id.example",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "OAUTH_ISSUER", "https://id.example")
    method, resource, forward_oauth_token = getattr(
        request,
        "param",
        ("client_secret_basic", "https://llm.example/", True),
    )
    monkeypatch.setattr(settings, "OAUTH_CLIENT_AUTH_METHOD", method)
    monkeypatch.setattr(settings, "OAUTH_RESOURCE", resource)
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", forward_oauth_token)
    monkeypatch.setattr(
        settings, "GATEWAY_API_KEY", None if forward_oauth_token else "static-key"
    )
    get_chainlit_settings.cache_clear()
    oauth_client.oidc_client.cache_clear()
    provider = OIDCProvider(auth_method=method, resource=resource)
    oauth_client.oidc_client().client_kwargs["transport"] = httpx2.MockTransport(
        provider
    )
    monkeypatch.setattr(
        oauth_client,
        "AsyncOAuth2Client",
        partial(AsyncOAuth2Client, transport=httpx2.MockTransport(provider)),
    )
    sessions: dict[tuple[str, str], OAuthTokens] = {}
    persisted: dict[str, cl.User] = {}

    async def save(
        session_id: str, identifier: str, tokens: OAuthTokens, expires_at: float
    ) -> None:
        assert expires_at > time()
        sessions[session_id, identifier] = tokens

    async def token(session_id: str, identifier: str) -> str:
        try:
            return sessions[session_id, identifier].access_token
        except KeyError:
            raise OAuthLoginRequired() from None

    async def delete(session_id: str, identifier: str) -> OAuthTokens | None:
        return sessions.pop((session_id, identifier), None)

    async def create_user(user: cl.User) -> cl.User:
        persisted[user.identifier] = deepcopy(user)
        return persisted[user.identifier]

    layer = SimpleNamespace(
        create_user=AsyncMock(side_effect=create_user),
        get_user=AsyncMock(side_effect=persisted.get),
    )
    monkeypatch.setattr(chainlit.data, "_data_layer", layer)
    monkeypatch.setattr(chainlit.data, "_data_layer_initialized", True)
    monkeypatch.setattr(auth, "save_oauth_tokens", save)
    monkeypatch.setattr(auth, "access_token", token)
    monkeypatch.setattr(auth, "delete_oauth_session", delete)
    app = FastAPI()
    auth.configure_auth(app)
    mount_chainlit(
        app=app,
        target=str(Path(__file__).parents[2] / "src/lgos_chainlit/simple.py"),
        path="",
    )
    gateway_authorizations: list[str] = []

    def catalog(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/model/info"
        gateway_authorizations.append(request.headers["Authorization"])
        return httpx.Response(200, json={"object": "list", "data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(catalog)) as gateway:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=gateway),
        )
        try:
            yield OAuthApp(app, provider, sessions, persisted, gateway_authorizations)
        finally:
            get_chainlit_settings.cache_clear()
            oauth_client.oidc_client.cache_clear()


@pytest.mark.parametrize(
    ("prompt", "provider_prompt", "expected"),
    [
        (None, None, None),
        ("login", None, "login"),
        ("login", "select_account", "select_account"),
    ],
)
async def test_authorization_uses_native_prompt_settings(
    oauth_app: OAuthApp,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str | None,
    provider_prompt: str | None,
    expected: str | None,
) -> None:
    for name, value in {
        "OAUTH_PROMPT": prompt,
        "OAUTH_GENERIC_PROMPT": provider_prompt,
    }.items():
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    async with oauth_app.browser() as browser:
        params = await oauth_app.start(browser, "unused")
    assert params.get("prompt") == ([expected] if expected is not None else None)


@pytest.mark.parametrize(
    "oauth_app",
    [
        ("client_secret_post", "https://llm.example/", True),
        ("client_secret_basic", None, True),
    ],
    indirect=True,
    ids=["post-with-resource", "basic-without-resource"],
)
async def test_pkce_login_keeps_grants_server_side_and_separates_same_user_sessions(
    oauth_app: OAuthApp,
) -> None:
    async with (
        oauth_app.browser() as first,
        oauth_app.browser() as second,
    ):
        for browser, code in ((first, "first"), (second, "second")):
            response = await oauth_app.login(browser, code)
            assert (
                response.status_code == 302
                and "success=true" in response.headers["location"]
            )
            cookie = response.headers.get_list("set-cookie")[0]
            assert (
                "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
            )
            assert "access-first" not in str(browser.cookies) and "refresh-" not in str(
                browser.cookies
            )
            assert "lgos_oauth_state" not in browser.cookies
        first_jwt = first.cookies["access_token"]
        assert first_jwt != second.cookies["access_token"]
        for browser, code in ((first, "first"), (second, "second")):
            user = await browser.get("/user")
            assert user.json()["metadata"] == {"provider": "generic"}
            oauth_app.gateway_authorizations.clear()
            assert (await browser.get("/project/settings")).status_code == 200
            assert set(oauth_app.gateway_authorizations) == {f"Bearer access-{code}"}
        assert (await first.post("/logout")).status_code == 200
        assert oauth_app.provider.revocations == ["refresh-first"]
        assert len(oauth_app.sessions) == 1
        assert "access_token" not in first.cookies
        assert (await first.get("/user")).status_code == 401
        # Native UI JWTs are not denylisted, but cannot recover deleted gateway grants.
        first.cookies.set("access_token", first_jwt)
        assert (await first.get("/user")).status_code == 200
        oauth_app.gateway_authorizations.clear()
        response = await first.get("/project/settings")
        assert response.status_code == 200 and response.json()["chatProfiles"] == []
        assert not oauth_app.gateway_authorizations
        assert (await second.get("/user")).status_code == 200
        assert (await second.get("/project/settings")).status_code == 200
        assert set(oauth_app.gateway_authorizations) == {"Bearer access-second"}
    assert oauth_app.persisted["alice"].metadata == {"provider": "generic"}


@pytest.mark.parametrize(
    "oauth_app",
    [("client_secret_basic", None, False)],
    indirect=True,
)
async def test_oidc_login_can_use_a_static_gateway_key(oauth_app: OAuthApp) -> None:
    async with oauth_app.browser() as browser:
        response = await oauth_app.login(browser, "static")
        assert response.status_code == 302
        assert "success=true" in response.headers["location"]
        assert not oauth_app.sessions

        user = await browser.get("/user")
        assert user.json()["metadata"] == {"provider": "generic"}
        assert (await browser.get("/project/settings")).status_code == 200
        assert oauth_app.gateway_authorizations == ["Bearer static-key"]

        assert (await browser.post("/logout")).status_code == 200
        assert not oauth_app.provider.revocations
        assert "access_token" not in browser.cookies


async def test_unsupported_client_auth_fails_before_authorization(
    oauth_app: OAuthApp,
) -> None:
    oauth_app.provider.auth_method = "private_key_jwt"
    async with oauth_app.browser() as browser:
        with pytest.raises(ValueError, match="client authentication method"):
            await browser.get("/auth/oauth/generic")
    assert not oauth_app.sessions


async def test_oidc_requires_a_validated_id_token_even_with_userinfo(
    oauth_app: OAuthApp,
) -> None:
    def provider(request: httpx2.Request) -> httpx2.Response:
        response = oauth_app.provider(request)
        if request.url.path == "/token":
            token = response.json()
            token.pop("id_token")
            token["userinfo"] = {"sub": "unverified-subject"}
            return httpx2.Response(200, json=token)
        return response

    oauth_client.oidc_client().client_kwargs["transport"] = httpx2.MockTransport(
        provider
    )
    async with oauth_app.browser() as browser:
        response = await oauth_app.login(browser, "no-id-token")
        assert "error=" in response.headers["location"]
        assert "access_token" not in browser.cookies
    assert not oauth_app.sessions
    assert not oauth_app.persisted


async def test_login_keeps_authlibs_token_expiration(oauth_app: OAuthApp) -> None:
    expires_at = int(time()) + 120
    oauth_app.provider.token_override = {"expires_at": expires_at}
    async with oauth_app.browser() as browser:
        response = await oauth_app.login(browser, "expiry")
        assert "success=true" in response.headers["location"]
    stored = next(iter(oauth_app.sessions.values()))
    assert stored.model_dump()["expires_at"] == expires_at


@pytest.mark.parametrize(
    "failure",
    ["state", "cookie", "nonce", "issuer", "audience", "expired", "signature"],
)
async def test_oidc_rejects_invalid_callback_without_creating_session(
    oauth_app: OAuthApp, failure: str
) -> None:
    async with oauth_app.browser() as browser:
        params = await oauth_app.start(browser, "invalid")
        state = params["state"][0]
        if failure == "state":
            state = "wrong-state"
        elif failure == "cookie":
            browser.cookies.clear()
        elif failure == "signature":
            oauth_app.provider.invalid_signature = True
        else:
            oauth_app.provider.claims_override = {
                "nonce": {"nonce": "wrong-nonce"},
                "issuer": {"iss": "https://evil.example"},
                "audience": {"aud": "another-client"},
                "expired": {"exp": int(time()) - 60},
            }[failure]
        response = await browser.get(
            "/auth/oauth/generic/callback", params={"code": "invalid", "state": state}
        )
        assert "error=" in response.headers["location"]
        assert "access_token" not in browser.cookies
        assert not oauth_app.sessions
        if failure in ("state", "cookie"):
            assert oauth_app.provider.exchanges == 0


async def test_logout_stays_local_when_provider_revocation_fails(
    oauth_app: OAuthApp,
) -> None:
    oauth_app.provider.revocation_status = 503
    async with oauth_app.browser() as browser:
        await oauth_app.login(browser, "first")
        assert (await browser.post("/logout")).status_code == 200
        assert not oauth_app.sessions
        assert "access_token" not in browser.cookies


async def test_logout_preserves_native_callback_response(
    oauth_app: OAuthApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth.config.code, "on_logout", None)

    @cl.on_logout
    def on_logout(request: Request, response: Response) -> dict[str, bool]:
        assert request.url.path == "/logout"
        assert not oauth_app.sessions
        response.delete_cookie("ui-preference")
        response.headers["X-Logout-Hook"] = "called"
        return {"signed_out": True}

    async with oauth_app.browser() as browser:
        await oauth_app.login(browser, "first")
        browser.cookies.set("ui-preference", "compact", domain="chat.example", path="/")
        response = await browser.post("/logout")
        assert response.status_code == 200
        assert response.json() == {"signed_out": True}
        assert response.headers["X-Logout-Hook"] == "called"
        assert "access_token" not in browser.cookies
        assert "ui-preference" not in browser.cookies
    assert oauth_app.provider.revocations == ["refresh-first"]


async def test_new_login_replaces_only_that_browsers_previous_session(
    oauth_app: OAuthApp,
) -> None:
    async with oauth_app.browser() as browser:
        await oauth_app.login(browser, "first")
        old_jwt = browser.cookies["access_token"]
        await oauth_app.login(browser, "second")
        assert len(oauth_app.sessions) == 1
        assert next(iter(oauth_app.sessions.values())).access_token == "access-second"
        assert (await browser.get("/user")).status_code == 200
        assert (await browser.get("/project/settings")).status_code == 200
        assert set(oauth_app.gateway_authorizations) == {"Bearer access-second"}
        browser.cookies.clear()
        browser.cookies.set("access_token", old_jwt)
        oauth_app.gateway_authorizations.clear()
        response = await browser.get("/project/settings")
        assert response.status_code == 200 and response.json()["chatProfiles"] == []
        assert not oauth_app.gateway_authorizations
