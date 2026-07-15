from pathlib import Path

import pytest
from demo.api.settings import settings
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.status import HTTP_200_OK

CHAINLIT_TARGET = Path("demo/ui/chainlit_ui/simple.py").resolve().as_posix()


@pytest.mark.anyio
async def test_mock_chainlit_login_returns_the_demo_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET",
        "test-chainlit-secret-with-at-least-32-bytes",
    )
    monkeypatch.setattr(settings, "CHAINLIT_LOGIN_TYPE", "mock")

    from chainlit.utils import mount_chainlit  # noqa: PLC0415

    app = FastAPI()
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
    assert user_response.json() == {
        "identifier": "demo-user",
        "metadata": {"provider": "mock"},
        "display_name": "Demo User",
    }
