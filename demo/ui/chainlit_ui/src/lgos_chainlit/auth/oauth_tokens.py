"""Encrypted, expiring OAuth grants, isolated by browser login in PostgreSQL."""

from time import time

import asyncpg
from authlib.integrations.base_client import OAuthError
from chainlit.data import get_data_layer
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lgos_chainlit.auth.oauth_client import oidc_metadata, token_client
from lgos_chainlit.settings import settings


class OAuthTokens(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str | None = Field(default=None, min_length=1, repr=False)
    # Authlib normalizes the provider's expires_in/expires_at at token receipt.
    expires_at: float = Field(gt=0, allow_inf_nan=False)


class OAuthLoginRequired(OpenAIError):
    def __init__(self) -> None:
        super().__init__("Gateway authorization expired. Log out and sign in again.")


def _cipher() -> MultiFernet:
    return MultiFernet(
        [Fernet(key.get_secret_value()) for key in settings.OAUTH_ENCRYPTION_KEYS]
    )


async def _pool() -> asyncpg.Pool:
    layer = get_data_layer()
    if not isinstance(layer, ChainlitDataLayer):
        raise RuntimeError("OAuth token forwarding requires Chainlit PostgreSQL.")
    await layer.connect()
    assert layer.pool is not None
    return layer.pool


async def initialize_oauth_storage() -> None:
    pool = await _pool()
    async with pool.acquire() as connection, connection.transaction():
        # IF NOT EXISTS alone can race when multiple workers start.
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext('lgos_chainlit_oauth_sessions'))"
        )
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS lgos_chainlit_oauth_sessions (
                session_id TEXT PRIMARY KEY,
                identifier TEXT NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL,
                tokens BYTEA NOT NULL
            )
        """)
        await connection.execute(
            "DELETE FROM lgos_chainlit_oauth_sessions WHERE expires_at <= $1", time()
        )


async def save_oauth_tokens(
    session_id: str, identifier: str, tokens: OAuthTokens, expires_at: float
) -> None:
    pool = await _pool()
    await pool.execute(
        "DELETE FROM lgos_chainlit_oauth_sessions WHERE expires_at <= $1", time()
    )
    await pool.execute(
        "INSERT INTO lgos_chainlit_oauth_sessions (session_id, identifier, expires_at, tokens) VALUES ($1, $2, $3, $4)",
        session_id,
        identifier,
        expires_at,
        _cipher().encrypt(tokens.model_dump_json().encode()),
    )


def _decrypt(encrypted: bytes | None) -> OAuthTokens:
    if encrypted is None:
        raise OAuthLoginRequired()
    try:
        return OAuthTokens.model_validate_json(_cipher().decrypt(encrypted))
    except (InvalidToken, ValidationError):
        raise OAuthLoginRequired() from None


async def delete_oauth_session(session_id: str, identifier: str) -> OAuthTokens | None:
    """Commit local invalidation before attempting any provider revocation."""
    pool = await _pool()
    encrypted = await pool.fetchval(
        "DELETE FROM lgos_chainlit_oauth_sessions WHERE session_id = $1 AND identifier = $2 RETURNING tokens",
        session_id,
        identifier,
    )
    try:
        return _decrypt(encrypted)
    except OAuthLoginRequired:
        return None


async def access_token(session_id: str, identifier: str) -> str:
    pool = await _pool()
    query = "SELECT tokens FROM lgos_chainlit_oauth_sessions WHERE session_id = $1 AND identifier = $2 AND expires_at > $3"
    tokens = _decrypt(await pool.fetchval(query, session_id, identifier, time()))
    if tokens.expires_at > time() + 30:
        return tokens.access_token

    async with pool.acquire() as connection, connection.transaction():
        await connection.execute("SET LOCAL lock_timeout = '10s'")
        # Re-read under a lock only for refresh, across tabs and workers.
        tokens = _decrypt(
            await connection.fetchval(
                query + " FOR UPDATE", session_id, identifier, time()
            )
        )
        if tokens.expires_at > time() + 30:
            return tokens.access_token
        if tokens.refresh_token is None:
            raise OAuthLoginRequired()
        metadata = await oidc_metadata()
        try:
            async with token_client() as client:
                result = await client.refresh_token(
                    metadata["token_endpoint"],
                    refresh_token=tokens.refresh_token,
                    resource=settings.OAUTH_RESOURCE,
                )
        except OAuthError as exc:
            if exc.error in ("invalid_grant", "invalid_token"):
                raise OAuthLoginRequired() from None
            raise
        refreshed = OAuthTokens.model_validate(result)
        if refreshed.refresh_token is None:
            refreshed.refresh_token = tokens.refresh_token
        await connection.execute(
            "UPDATE lgos_chainlit_oauth_sessions SET tokens = $2 WHERE session_id = $1",
            session_id,
            _cipher().encrypt(refreshed.model_dump_json().encode()),
        )
        return refreshed.access_token
