from typing import Any

import pytest
from botocore.exceptions import ClientError
from httpx2 import ASGITransport, AsyncClient

from lgos_files_api.app import create_app
from lgos_files_api.core.settings import settings
from tests.support import BUCKET


@pytest.mark.parametrize(
    ("attribute", "environment_name"),
    [
        ("BUCKET", "DEMO_API_FILES_BUCKET"),
        ("AWS_ACCESS_KEY_ID", "DEMO_API_FILES_AWS_ACCESS_KEY_ID"),
        ("AWS_SECRET_ACCESS_KEY", "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY"),
        ("AWS_DEFAULT_REGION", "DEMO_API_FILES_AWS_DEFAULT_REGION"),
    ],
)
def test_files_service_requires_its_s3_settings(
    s3: Any,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    environment_name: str,
) -> None:
    monkeypatch.setattr(settings, attribute, None)

    with pytest.raises(RuntimeError, match=environment_name):
        create_app()


async def test_unexpected_error_leaves_its_traceback_to_the_server(
    s3: Any, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app()

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        s3.delete_bucket(Bucket=BUCKET)
        with pytest.raises(ClientError, match="NoSuchBucket"):
            await client.get("/v1/files/file-" + "1" * 32)

    (record,) = [r for r in caplog.records if r.name == "lgos_files_api.core.errors"]
    assert record.exc_info is None
