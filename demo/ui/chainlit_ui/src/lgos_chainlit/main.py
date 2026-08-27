import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chainlit.data import get_data_layer
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from lgos_chainlit.settings import get_chainlit_settings, settings

os.environ.setdefault(
    "AWS_CONFIG_FILE",
    Path(__file__).with_name("aws_config").as_posix(),
)
get_chainlit_settings()


async def _close_chainlit_data_layer() -> None:
    data_layer = get_data_layer()
    if not isinstance(data_layer, ChainlitDataLayer):
        return
    if isinstance(data_layer.storage_client, S3StorageClient):
        # Chainlit 2.11.1 incorrectly awaits boto3's synchronous close method.
        data_layer.storage_client.client.close()
        data_layer.storage_client = None
    await data_layer.close()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await _close_chainlit_data_layer()


app = FastAPI(lifespan=lifespan)

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
