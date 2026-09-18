"""Demo authentication policy and gateway credential tests."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import chainlit as cl
import httpx2
import pytest
from chainlit.auth import create_jwt
from chainlit.context import ChainlitContext, context_var
from chainlit.session import WebsocketSession
from chainlit_utils.sso.tokens import OAuthLoginRequired, OAuthTokenStore
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from starlette.status import HTTP_200_OK

from lgos_chainlit import auth, clients
from lgos_chainlit.settings import settings

CHAINLIT_TARGET = (
    Path(__file__).parents[1] / "src" / "lgos_chainlit" / "chat.py"
).as_posix()


@pytest.fixture
def delegated_store(monkeypatch: pytest.MonkeyPatch) -> Mock:
    store = Mock(spec=OAuthTokenStore)
    monkeypatch.setattr(auth, "token_store", lambda: store)
    return store


@asynccontextmanager
async def chat_session(user: cl.User, token: str) -> AsyncIterator[WebsocketSession]:
    session = WebsocketSession(
        id=uuid4().hex,
        socket_id=uuid4().hex,
        emit=AsyncMock(),
        emit_call=AsyncMock(),
        user_env={},
        client_type="webapp",
        user=user,
        token=token,
    )
    context_token = context_var.set(ChainlitContext(session))
    try:
        yield session
    finally:
        context_var.reset(context_token)
        await session.delete()


async def test_mock_chainlit_login_returns_the_demo_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET",
        "test-chainlit-secret-with-at-least-32-bytes",
    )
    monkeypatch.setattr(settings, "LOGIN_TYPE", "mock")

    from chainlit.utils import mount_chainlit

    app = FastAPI()
    auth.configure_auth(app)
    mount_chainlit(
        app=app,
        target=CHAINLIT_TARGET,
        path="",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        login_response = await client.post(
            "/login",
            data={"username": "anything", "password": "anything"},
        )
        user_response = await client.get("/user")

    assert login_response.status_code == HTTP_200_OK
    assert login_response.json() == {"success": True}
    assert user_response.status_code == HTTP_200_OK
    user = user_response.json()
    assert {key: user[key] for key in ("identifier", "metadata", "display_name")} == {
        "identifier": "demo-user",
        "metadata": {"provider": "mock"},
        "display_name": "Demo User",
    }


@pytest.mark.parametrize("forward_tokens", [False, True])
def test_oauth_configuration_maps_the_demo_login_policy(
    monkeypatch: pytest.MonkeyPatch,
    forward_tokens: bool,
) -> None:
    native = SimpleNamespace(
        OAUTH_GENERIC_NAME="generic",
        CHAINLIT_URL="https://chat.example",
        CHAINLIT_AUTH_SECRET="test-signing-secret-with-at-least-32-bytes",
    )
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", forward_tokens)
    monkeypatch.setattr(auth, "get_chainlit_settings", lambda: native)
    oidc = Mock()
    store = Mock()
    oidc_factory = Mock(return_value=oidc)
    store_factory = Mock(return_value=store)
    integration = Mock()
    integration_factory = Mock(return_value=integration)
    monkeypatch.setattr(auth, "oidc_client", oidc_factory)
    monkeypatch.setattr(auth, "token_store", store_factory)
    monkeypatch.setattr(auth, "ChainlitOAuth", integration_factory)
    app = FastAPI()

    auth.configure_auth(app)

    integration_factory.assert_called_once_with(
        provider_id="generic",
        chainlit_url="https://chat.example",
        auth_secret="test-signing-secret-with-at-least-32-bytes",
        oidc=oidc,
        provider_env=(
            "OAUTH_GENERIC_CLIENT_ID",
            "OAUTH_GENERIC_CLIENT_SECRET",
            "DEMO_CHAINLIT_OAUTH_ISSUER",
        ),
        token_store=store if forward_tokens else None,
        session_claim=auth.SESSION_CLAIM,
        state_cookie="lgos_oauth_state",
    )
    integration.configure.assert_called_once_with(app)
    if forward_tokens:
        store_factory.assert_called_once_with()
    else:
        store_factory.assert_not_called()


async def test_missing_delegated_user_preserves_login_error_through_sdk(
    monkeypatch: pytest.MonkeyPatch,
    delegated_store: Mock,
) -> None:
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", True)

    def gateway(_request: httpx2.Request) -> httpx2.Response:
        pytest.fail("Unauthenticated request reached the gateway")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(gateway)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        with pytest.raises(OAuthLoginRequired, match="sign in again"):
            await clients.list_models()


async def test_delegated_chat_uses_new_credentials_and_stops_after_logout(
    monkeypatch: pytest.MonkeyPatch,
    delegated_store: Mock,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", True)
    user = cl.User(identifier="alice", metadata={auth.SESSION_CLAIM: "session"})
    current_token: str | None = "access-before-refresh"
    observed: list[str] = []

    async def token(session_id: str, identifier: str) -> str:
        assert (session_id, identifier) == ("session", "alice")
        if current_token is None:
            raise OAuthLoginRequired()
        return current_token

    def gateway(request: httpx2.Request) -> httpx2.Response:
        observed.append(request.headers["Authorization"])
        return httpx2.Response(200, json={"id": "resp_test", "output": []})

    delegated_store.access_token = token
    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(gateway)) as http,
        chat_session(user, create_jwt(user)),
    ):
        client = clients.openai_client.with_options(http_client=http)
        await client.responses.create(model="graph", input="first message")
        current_token = "access-after-refresh"
        await client.responses.create(model="graph", input="next message")
        current_token = None
        with pytest.raises(OAuthLoginRequired):
            await client.responses.create(model="graph", input="after logout")
    assert observed == ["Bearer access-before-refresh", "Bearer access-after-refresh"]


@pytest.mark.parametrize("login_type", ["mock", "oauth"])
async def test_gateway_uses_its_static_key_without_a_user_session(
    monkeypatch: pytest.MonkeyPatch, login_type: str
) -> None:
    monkeypatch.setattr(settings, "LOGIN_TYPE", login_type)
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", False)
    monkeypatch.setattr(settings, "OPENAI_GATEWAY_API_KEY", "static-key")

    def gateway(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == "Bearer static-key"
        assert request.url.path == "/model/info"
        return httpx2.Response(200, json={"object": "list", "data": []})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(gateway)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        assert await clients.list_models() == []
