from collections.abc import AsyncIterator, Iterator

import boto3
import pytest
from moto import mock_aws
from openai import AsyncOpenAI

from lgos_files_api.app import create_app
from lgos_files_api.core.settings import settings
from tests.support import BUCKET, serve


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the supported backend."""
    return "asyncio"


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Yield an in-process S3 holding the service's empty bucket, and configure it."""
    for name, value in {
        "BUCKET": BUCKET,
        "S3_ENDPOINT": None,
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setattr(settings, name, value)
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
async def files_client(s3: object) -> AsyncIterator[AsyncOpenAI]:
    async with serve(create_app()) as client:
        yield client
