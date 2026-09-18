import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chainlit.config import config
from chainlit.data import get_data_layer
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from chainlit.utils import mount_chainlit
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

if settings.UI_FILE == "simple" and not settings.ENABLE_OAUTH_TOKEN_FORWARDING:
    assert settings.OPENAI_GATEWAY_API_KEY is not None
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
        # Chainlit 2.12.0 incorrectly awaits boto3's synchronous close method.
        data_layer.storage_client.client.close()
        data_layer.storage_client = None
    await data_layer.close()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        if settings.ENABLE_OAUTH_TOKEN_FORWARDING:
            await token_store().initialize()
        yield
    finally:
        await gateway_http_client.aclose()
        await _close_chainlit_data_layer()


app = FastAPI(lifespan=lifespan)
configure_auth(app)

CHAINLIT_UI_PATH = f"{settings.UI_FILE}.py"

mount_chainlit(
    app=app,
    target=Path(__file__).parent.joinpath(CHAINLIT_UI_PATH).absolute().as_posix(),
    path="",
)


def run() -> None:
    """Run the Chainlit application."""
    import uvicorn

    uvicorn.run("lgos_chainlit.main:app", host="0.0.0.0", port=5000)
