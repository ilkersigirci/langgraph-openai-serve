"""S3 file repository tests."""

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import pytest
from botocore.client import BaseClient
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from lgos_files_api.contracts import FileUpload, StoredFileNotFoundError
from lgos_files_api.s3 import S3FileRepository

BUCKET = "files"
FILE_ID = "file-abc123"
KEY = f"openai-files/{FILE_ID}"
CREATED_AT = 123


@pytest.fixture
def s3_client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url="https://s3.example.com",
        region_name="eu-west-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def test_upload_uses_boto3_transfer_and_returns_openai_file(
    s3_client: BaseClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"payload"
    upload = FileUpload(
        body=BytesIO(payload),
        size=len(payload),
        filename="report.pdf",
        content_type="application/pdf",
        purpose="user_data",
    )
    repository = S3FileRepository(s3_client, bucket=BUCKET)
    monkeypatch.setattr("lgos_files_api.s3.time", lambda: CREATED_AT)

    with Stubber(s3_client) as stubber:
        stubber.add_response(
            "put_object",
            {},
            {
                "Body": ANY,
                "Bucket": BUCKET,
                "ChecksumAlgorithm": ANY,
                "ContentType": "application/pdf",
                "Key": ANY,
                "Metadata": {
                    "created-at": str(CREATED_AT),
                    "filename": "cmVwb3J0LnBkZg==",
                    "purpose": "user_data",
                },
            },
        )
        stored = repository.create(upload)

    assert stored.id.startswith("file-")
    assert stored.filename == "report.pdf"
    assert stored.bytes == len(payload)
    assert stored.created_at == CREATED_AT


def test_list_orders_and_pages_before_loading_metadata() -> None:
    times = {
        "file-oldest": 1,
        "file-old": 2,
        "file-mid": 3,
        "file-new": 4,
    }
    contents = [
        {
            "Key": f"openai-files/{file_id}",
            "LastModified": datetime.fromtimestamp(created_at, tz=UTC),
        }
        for file_id, created_at in times.items()
    ]
    paginator = SimpleNamespace(
        paginate=Mock(return_value=[{"Contents": list(reversed(contents))}])
    )
    client = Mock()
    client.get_paginator.return_value = paginator

    def head_object(**kwargs: str) -> dict[str, object]:
        assert kwargs["Bucket"] == BUCKET
        file_id = kwargs["Key"].rsplit("/", 1)[-1]
        created_at = times[file_id]
        return {
            "ContentLength": created_at,
            "LastModified": datetime.fromtimestamp(created_at, tz=UTC),
            "Metadata": {
                "created-at": str(created_at),
                "filename": "ZmlsZS5iaW4=",
                "purpose": "user_data",
            },
        }

    client.head_object.side_effect = head_object
    repository = S3FileRepository(client, bucket=BUCKET)

    page = repository.list_files(
        after=None,
        limit=2,
        order="desc",
        purpose=None,
    )

    assert [file.id for file in page.data] == ["file-new", "file-mid"]
    assert page.has_more is True
    assert [call.kwargs["Key"] for call in client.head_object.call_args_list] == [
        "openai-files/file-new",
        "openai-files/file-mid",
        "openai-files/file-old",
    ]


def test_retrieve_delete_and_content_use_stored_object(s3_client: BaseClient) -> None:
    payload = b"payload"
    metadata = {
        "created-at": "1",
        "filename": "cmVwb3J0LnBkZg==",
        "purpose": "user_data",
    }
    head = {
        "ContentLength": len(payload),
        "ContentType": "application/pdf",
        "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
        "Metadata": metadata,
    }
    repository = S3FileRepository(s3_client, bucket=BUCKET)

    with Stubber(s3_client) as stubber:
        stubber.add_response("head_object", head, {"Bucket": BUCKET, "Key": KEY})
        stored = repository.retrieve(FILE_ID)
        stubber.add_response(
            "get_object",
            {**head, "Body": StreamingBody(BytesIO(payload), len(payload))},
            {"Bucket": BUCKET, "Key": KEY},
        )
        download = repository.content(FILE_ID)
        assert b"".join(download.body) == payload
        stubber.add_response("head_object", head, {"Bucket": BUCKET, "Key": KEY})
        stubber.add_response("delete_object", {}, {"Bucket": BUCKET, "Key": KEY})
        deleted = repository.delete(FILE_ID)

    assert stored.filename == "report.pdf"
    assert download.content_type == "application/pdf"
    assert deleted.id == FILE_ID
    assert deleted.deleted is True


def test_missing_s3_object_becomes_stored_file_not_found(
    s3_client: BaseClient,
) -> None:
    repository = S3FileRepository(s3_client, bucket=BUCKET)

    with Stubber(s3_client) as stubber:
        stubber.add_client_error(
            "head_object",
            service_error_code="NoSuchKey",
            http_status_code=404,
            expected_params={"Bucket": BUCKET, "Key": KEY},
        )
        with pytest.raises(StoredFileNotFoundError):
            repository.retrieve(FILE_ID)
