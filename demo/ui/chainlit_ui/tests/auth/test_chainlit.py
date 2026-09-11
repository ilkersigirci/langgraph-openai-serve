from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import chainlit as cl
import httpx
import pytest
from chainlit.auth import create_jwt
from chainlit.context import ChainlitContext, context_var
from chainlit.session import WebsocketSession
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.status import HTTP_200_OK

from lgos_chainlit.auth import chainlit as auth
from lgos_chainlit.auth.oauth_tokens import OAuthLoginRequired
from lgos_chainlit.settings import settings
from lgos_chainlit.utils import clients

CHAINLIT_TARGET = (
    Path(__file__).parents[2] / "src" / "lgos_chainlit" / "simple.py"
).as_posix()


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


@pytest.mark.parametrize("surface", ["http", "chat"])
async def test_concurrent_requests_keep_each_users_gateway_credentials_isolated(
    monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", True)
    observed: dict[str, str] = {}
    both_requested = anyio.Event()
    pending = 0

    async def token(session_id: str, identifier: str) -> str:
        assert session_id == f"session-{identifier}"
        nonlocal pending
        pending += 1
        if pending == 2:
            both_requested.set()
        await both_requested.wait()
        return f"access-{identifier}"

    def gateway(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["Authorization"]
        subject = request.headers["X-Test-Subject"]
        observed[subject] = authorization
        return httpx.Response(200, json={"object": "list", "data": []})

    monkeypatch.setattr(auth, "access_token", token)
    async with AsyncClient(transport=httpx.MockTransport(gateway)) as http:
        client = clients.openai_client.with_options(http_client=http)
        app = FastAPI()
        app.add_middleware(auth.GatewayRequestContextMiddleware)

        @app.get("/credential/{subject}")
        async def credential(subject: str) -> str:
            await client.models.list(extra_headers={"X-Test-Subject": subject})
            return "ok"

        async def request_credential(subject: str) -> None:
            user = cl.User(
                identifier=subject,
                metadata={auth.SESSION_CLAIM: f"session-{subject}"},
            )
            jwt = create_jwt(user)
            if surface == "http":
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="https://chat.example"
                ) as browser:
                    response = await browser.get(
                        f"/credential/{subject}",
                        headers={"Authorization": f"Bearer {jwt}"},
                    )
                    assert response.status_code == 200
            else:
                async with chat_session(user, jwt):
                    await client.models.list(extra_headers={"X-Test-Subject": subject})

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(request_credential, "alice")
                group.start_soon(request_credential, "bob")
    assert observed == {
        "alice": "Bearer access-alice",
        "bob": "Bearer access-bob",
    }


async def test_missing_delegated_user_preserves_login_error_through_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", True)

    def gateway(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Unauthenticated request reached the gateway")

    async with AsyncClient(transport=httpx.MockTransport(gateway)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        with pytest.raises(OAuthLoginRequired, match="sign in again"):
            await clients.list_models()


async def test_delegated_chat_uses_new_credentials_and_stops_after_logout(
    monkeypatch: pytest.MonkeyPatch,
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

    def gateway(request: httpx.Request) -> httpx.Response:
        observed.append(request.headers["Authorization"])
        return httpx.Response(200, json={"id": "resp_test", "output": []})

    monkeypatch.setattr(auth, "access_token", token)
    async with (
        AsyncClient(transport=httpx.MockTransport(gateway)) as http,
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


async def test_delegated_chat_rejects_a_credential_bound_to_another_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    monkeypatch.setattr(settings, "LOGIN_TYPE", "oauth")
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", True)
    alice_token = create_jwt(
        cl.User(identifier="alice", metadata={auth.SESSION_CLAIM: "alice-session"})
    )
    async with chat_session(cl.User(identifier="bob"), alice_token):
        with pytest.raises(OAuthLoginRequired):
            await auth.gateway_credential()


@pytest.mark.parametrize("login_type", ["mock", "oauth"])
async def test_gateway_uses_its_static_key_without_a_user_session(
    monkeypatch: pytest.MonkeyPatch, login_type: str
) -> None:
    monkeypatch.setattr(settings, "LOGIN_TYPE", login_type)
    monkeypatch.setattr(settings, "ENABLE_OAUTH_TOKEN_FORWARDING", False)
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "static-key")

    def gateway(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer static-key"
        assert request.url.path == "/model/info"
        return httpx.Response(200, json={"object": "list", "data": []})

    async with AsyncClient(transport=httpx.MockTransport(gateway)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        assert await clients.list_models() == []
