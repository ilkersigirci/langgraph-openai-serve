from pathlib import Path

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from lgos_chainlit.settings import get_chainlit_settings, settings

get_chainlit_settings()

app = FastAPI()

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
