from pathlib import Path

from chainlit.utils import mount_chainlit
from demo.api.settings import get_chainlit_settings, settings
from fastapi import FastAPI

get_chainlit_settings()

app = FastAPI()

CHAINLIT_UI_PATH = f"{settings.CHAINLIT_UI_FILE}.py"

mount_chainlit(
    app=app,
    target=Path(__file__).parent.joinpath(CHAINLIT_UI_PATH).absolute().as_posix(),
    path="",
)
