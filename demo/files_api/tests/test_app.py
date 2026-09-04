from unittest.mock import Mock

import pytest

from lgos_files_api import app as app_module


def configure_files_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module.settings, "BUCKET", "files")
    monkeypatch.setattr(app_module.settings, "S3_ENDPOINT", "https://s3.test")
    monkeypatch.setattr(app_module.settings, "AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setattr(app_module.settings, "AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(app_module.settings, "AWS_DEFAULT_REGION", "eu-west-1")


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
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    environment_name: str,
) -> None:
    configure_files_settings(monkeypatch)
    monkeypatch.setattr(app_module.settings, attribute, None)

    with pytest.raises(RuntimeError, match=environment_name):
        app_module.create_s3_app()


async def test_files_service_exposes_routes_and_closes_boto3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3_client = Mock()
    boto3_client = Mock(return_value=s3_client)
    configure_files_settings(monkeypatch)
    monkeypatch.setattr(app_module.boto3, "client", boto3_client)

    app = app_module.create_s3_app()

    boto3_client.assert_called_once_with(
        "s3",
        endpoint_url="https://s3.test",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
        region_name="eu-west-1",
    )
    assert "/health" in {getattr(route, "path", None) for route in app.routes}
    async with app.router.lifespan_context(app):
        pass
    s3_client.close.assert_called_once_with()
