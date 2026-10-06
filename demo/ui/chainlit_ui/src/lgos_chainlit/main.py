import os
from collections.abc import AsyncGenerator, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from chainlit.config import config
from chainlit.data import get_data_layer
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from chainlit.utils import mount_chainlit
from chainlit_utils.db.schema import setup_chainlit_schema
from chainlit_utils.public_files import serve_public_files
from chainlit_utils.sessions import keep_restored_sessions
from fastapi import FastAPI

from lgos_chainlit.auth import configure_auth, token_store
from lgos_chainlit.clients import gateway, gateway_http_client
from lgos_chainlit.mcp import mcp_gateway_config
from lgos_chainlit.settings import get_chainlit_settings, settings

os.environ.setdefault(
    "AWS_CONFIG_FILE",
    Path(__file__).with_name("aws_config").as_posix(),
)
get_chainlit_settings()
keep_restored_sessions()
serve_public_files()
config.features.audio.enabled = settings.AUDIO_STT_MODEL is not None

if not settings.ENABLE_OAUTH_TOKEN_FORWARDING:
    assert settings.OPENAI_GATEWAY_API_KEY is not None  # ruff: ignore[assert] - Pydantic settings validation already enforces this invariant.
    config.features.mcp.servers = [
        mcp_gateway_config(gateway, settings.OPENAI_GATEWAY_API_KEY)
    ]
else:
    config.features.mcp.enabled = False


async def _close_chainlit_data_layer() -> None:
    data_layer = get_data_layer()
    if not isinstance(data_layer, ChainlitDataLayer):
        return
    if isinstance(data_layer.storage_client, S3StorageClient):
        # Chainlit incorrectly awaits boto3's synchronous close method.
        data_layer.storage_client.client.close()
        data_layer.storage_client = None
    await data_layer.close()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        await setup_chainlit_schema(str(get_chainlit_settings().DATABASE_URL))
        if settings.ENABLE_OAUTH_TOKEN_FORWARDING:
            await token_store().initialize()
        yield
    finally:
        await gateway_http_client.aclose()
        await _close_chainlit_data_layer()


def _untraced(scope: MutableMapping[str, Any]) -> bool:
    # One Socket.IO connection carries a whole chat session. Leaving it out lets
    # each outbound gateway request start its own trace.
    path = scope["path"]
    return path.endswith("/health") or path.startswith("/ws/socket.io")


app = FastAPI(
    lifespan=lifespan,
    # FastAPI records requests through the global providers; export belongs to
    # `opentelemetry-instrument`, so FastAPI must not add its own.
    telemetry={"auto_configure": False, "exclude": _untraced},
)
configure_auth(app)

mount_chainlit(
    app=app,
    target=Path(__file__).with_name("chat.py").absolute().as_posix(),
    path="",
)


def run() -> None:
    """Run the Chainlit application."""
    import uvicorn

    uvicorn.run("lgos_chainlit.main:app", host="0.0.0.0", port=5000)
