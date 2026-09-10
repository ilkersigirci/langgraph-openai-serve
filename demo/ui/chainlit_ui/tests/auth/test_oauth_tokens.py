"""Credential persistence and refresh against an isolated PostgreSQL schema."""

import os
from base64 import b64decode
from collections.abc import AsyncIterator
from functools import partial
from time import time
from unittest.mock import AsyncMock
from urllib.parse import parse_qs
from uuid import uuid4

import anyio
import asyncpg
import httpx
import pytest
from authlib.integrations.httpx_client import AsyncOAuth2Client
from cryptography.fernet import Fernet
from pydantic import SecretStr

from lgos_chainlit.auth import oauth_client, oauth_tokens
from lgos_chainlit.auth.oauth_tokens import (
    OAuthLoginRequired,
    OAuthTokens,
    access_token,
    delete_oauth_session,
    initialize_oauth_storage,
    save_oauth_tokens,
)
from lgos_chainlit.settings import get_chainlit_settings, settings

pytestmark = pytest.mark.integration


@pytest.fixture
async def credential_database(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[asyncpg.Pool]:
    database_url = os.environ.get("TEST_CHAINLIT_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_CHAINLIT_DATABASE_URL to run PostgreSQL OAuth tests.")
    monkeypatch.setattr(
        settings, "OAUTH_ENCRYPTION_KEYS", [SecretStr(Fernet.generate_key().decode())]
    )
    monkeypatch.setattr(
        oauth_tokens,
        "oidc_metadata",
        AsyncMock(return_value={"token_endpoint": "https://id.example/token"}),
    )
    monkeypatch.setattr(settings, "OAUTH_RESOURCE", "https://llm.example/")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    monkeypatch.setenv("OAUTH_GENERIC_CLIENT_ID", "chainlit-client")
    monkeypatch.setenv("OAUTH_GENERIC_CLIENT_SECRET", "client-secret")
    get_chainlit_settings.cache_clear()
    schema = f"chainlit_oauth_test_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=2,
            server_settings={"search_path": schema},
        ) as pool:
            monkeypatch.setattr(oauth_tokens, "_pool", AsyncMock(return_value=pool))
            async with anyio.create_task_group() as group:
                group.start_soon(initialize_oauth_storage)
                group.start_soon(initialize_oauth_storage)
            yield pool
    finally:
        get_chainlit_settings.cache_clear()
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


async def test_tokens_are_encrypted_and_survive_reconnection(
    credential_database: asyncpg.Pool,
) -> None:
    await save_oauth_tokens(
        "alice-session",
        "alice",
        OAuthTokens(
            access_token="alice-access-secret",
            refresh_token="alice-refresh-secret",
            expires_at=time() + 3600,
        ),
        time() + 36000,
    )
    stored = await credential_database.fetchval(
        "SELECT tokens FROM lgos_chainlit_oauth_sessions WHERE identifier = 'alice'"
    )
    assert b"alice-access-secret" not in stored
    assert b"alice-refresh-secret" not in stored
    await credential_database.expire_connections()
    assert await access_token("alice-session", "alice") == "alice-access-secret"
    with pytest.raises(OAuthLoginRequired):
        await access_token("alice-session", "bob")


@pytest.mark.parametrize(
    ("rotate", "method", "resource"),
    [
        (True, "client_secret_post", "https://llm.example/"),
        (False, "client_secret_basic", None),
    ],
)
async def test_concurrent_refresh_is_serialized_and_keeps_latest_refresh_token(
    credential_database: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
    rotate: bool,
    method: str,
    resource: str | None,
) -> None:
    monkeypatch.setattr(settings, "OAUTH_CLIENT_AUTH_METHOD", method)
    monkeypatch.setattr(settings, "OAUTH_RESOURCE", resource)
    await save_oauth_tokens(
        "alice-session",
        "alice",
        OAuthTokens(
            access_token="expired",
            refresh_token="refresh-original",
            expires_at=1,
        ),
        time() + 36000,
    )
    requests: list[dict[str, list[str]]] = []
    refreshed: list[str] = []
    second_started = anyio.Event()

    async def oidc(request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode(), keep_blank_values=True)
        if method == "client_secret_basic":
            scheme, credentials = request.headers["authorization"].split()
            assert scheme == "Basic"
            assert b64decode(credentials).decode() == "chainlit-client:client-secret"
            assert "client_secret" not in form
        else:
            assert form["client_id"] == ["chainlit-client"]
            assert form["client_secret"] == ["client-secret"]
        assert form.get("resource") == ([resource] if resource else None)
        requests.append(form)
        await second_started.wait()
        result = {"access_token": "access-new", "expires_in": 3600}
        if rotate:
            result["refresh_token"] = "refresh-rotated"
        return httpx.Response(200, json=result)

    monkeypatch.setattr(
        oauth_client,
        "AsyncOAuth2Client",
        partial(AsyncOAuth2Client, transport=httpx.MockTransport(oidc)),
    )

    async def request_token(second: bool) -> None:
        if second:
            second_started.set()
        refreshed.append(await access_token("alice-session", "alice"))

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(request_token, False)
            group.start_soon(request_token, True)
    assert refreshed == ["access-new", "access-new"]
    assert len(requests) == 1
    assert requests[0]["grant_type"] == ["refresh_token"]
    assert requests[0]["refresh_token"] == ["refresh-original"]

    # Force the next request past the refreshed access token's expiry.
    monkeypatch.setattr(oauth_tokens, "time", lambda: time() + 7200)
    assert await access_token("alice-session", "alice") == "access-new"
    assert requests[1]["refresh_token"] == [
        "refresh-rotated" if rotate else "refresh-original"
    ]


@pytest.mark.parametrize(
    "failure",
    ["no-refresh-token", "revoked", "encryption-key-removed", "invalid-grant-data"],
)
async def test_unusable_credentials_require_login(
    credential_database: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    await save_oauth_tokens(
        "alice-session",
        "alice",
        OAuthTokens(
            access_token="expired",
            expires_at=1,
            refresh_token=None if failure == "no-refresh-token" else "revoked-refresh",
        ),
        time() + 36000,
    )
    if failure == "encryption-key-removed":
        monkeypatch.setattr(
            settings,
            "OAUTH_ENCRYPTION_KEYS",
            [SecretStr(Fernet.generate_key().decode())],
        )
    elif failure == "invalid-grant-data":
        encrypted = Fernet(
            settings.OAUTH_ENCRYPTION_KEYS[0].get_secret_value()
        ).encrypt(b"{}")
        await credential_database.execute(
            "UPDATE lgos_chainlit_oauth_sessions SET tokens = $1", encrypted
        )

    def oidc(_request: httpx.Request) -> httpx.Response:
        assert failure == "revoked"
        return httpx.Response(400, json={"error": "invalid_grant"})

    monkeypatch.setattr(
        oauth_tokens,
        "token_client",
        lambda: AsyncOAuth2Client(
            client_id="chainlit-client",
            client_secret="client-secret",
            token_endpoint_auth_method="client_secret_post",
            transport=httpx.MockTransport(oidc),
        ),
    )
    with pytest.raises(OAuthLoginRequired, match="Log out and sign in again"):
        await access_token("alice-session", "alice")
    await delete_oauth_session("alice-session", "alice")
    assert (
        await credential_database.fetchval(
            "SELECT count(*) FROM lgos_chainlit_oauth_sessions"
        )
        == 0
    )


async def test_same_user_sessions_have_independent_logout_and_expiry(
    credential_database: asyncpg.Pool,
) -> None:
    for session_id in ("first", "second"):
        await save_oauth_tokens(
            session_id,
            "alice",
            OAuthTokens(access_token=session_id, expires_at=time() + 3600),
            time() + 3600,
        )
    await delete_oauth_session("first", "alice")
    with pytest.raises(OAuthLoginRequired):
        await access_token("first", "alice")
    assert await access_token("second", "alice") == "second"
    assert await delete_oauth_session("second", "bob") is None
    await credential_database.execute(
        "UPDATE lgos_chainlit_oauth_sessions SET expires_at = 0 WHERE session_id = 'second'"
    )
    with pytest.raises(OAuthLoginRequired):
        await access_token("second", "alice")
    await initialize_oauth_storage()
    assert (
        await credential_database.fetchval(
            "SELECT count(*) FROM lgos_chainlit_oauth_sessions"
        )
        == 0
    )


async def test_key_rotation_reads_old_grants_and_encrypts_new_grants_with_primary_key(
    credential_database: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_key = settings.OAUTH_ENCRYPTION_KEYS[0]
    new_key = SecretStr(Fernet.generate_key().decode())
    tokens = OAuthTokens(access_token="secret-access", expires_at=time() + 3600)
    await save_oauth_tokens("old", "alice", tokens, time() + 3600)
    monkeypatch.setattr(settings, "OAUTH_ENCRYPTION_KEYS", [new_key, old_key])
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "a-completely-different-browser-signing-secret"
    )
    assert await access_token("old", "alice") == "secret-access"
    await save_oauth_tokens("new", "alice", tokens, time() + 3600)
    encrypted = await credential_database.fetchval(
        "SELECT tokens FROM lgos_chainlit_oauth_sessions WHERE session_id = 'new'"
    )
    assert b"secret-access" in Fernet(new_key.get_secret_value()).decrypt(encrypted)


async def test_logout_waits_for_refresh_and_removes_the_rotated_grant(
    credential_database: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    await save_oauth_tokens(
        "session",
        "alice",
        OAuthTokens(
            access_token="expired",
            refresh_token="old-refresh",
            expires_at=1,
        ),
        time() + 3600,
    )
    refreshing = anyio.Event()
    deleting = anyio.Event()
    removed: list[OAuthTokens | None] = []

    async def oidc(_request: httpx.Request) -> httpx.Response:
        refreshing.set()
        await deleting.wait()
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(
        oauth_tokens,
        "token_client",
        lambda: AsyncOAuth2Client(
            client_id="client", transport=httpx.MockTransport(oidc)
        ),
    )

    async def logout() -> None:
        await refreshing.wait()
        deleting.set()
        removed.append(await delete_oauth_session("session", "alice"))

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(access_token, "session", "alice")
            group.start_soon(logout)
    assert removed[0] is not None and removed[0].refresh_token == "new-refresh"
    with pytest.raises(OAuthLoginRequired):
        await access_token("session", "alice")
