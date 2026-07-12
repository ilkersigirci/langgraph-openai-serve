from pathlib import Path

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

app = FastAPI()

ui_path = "simple.py"

mount_chainlit(
    app=app,
    target=Path(__file__).parent.joinpath(ui_path).absolute().as_posix(),
    path="",
)
