"""Standalone OpenAI-compatible Files service backed by S3."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI

from lgos_files_api.api.files.extraction import Extractions
from lgos_files_api.api.files.service import FileService
from lgos_files_api.api.files.views import router as files_router
from lgos_files_api.api.health.views import router as health_router
from lgos_files_api.core.errors import configure_openai_error_handlers
from lgos_files_api.core.settings import settings
from lgos_files_api.storage import S3Bucket


def _required_setting(name: str, value: str | None) -> str:
    if value:
        return value
    msg = f"{name} is required by lgos-files-api."
    raise RuntimeError(msg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Run extractions while the app serves; close the boto3 client at shutdown."""
    try:
        async with app.state.extractions.run():
            yield
    finally:
        app.state.s3_client.close()


def create_app() -> FastAPI:
    """Create the Files service for the configured S3 bucket."""
    bucket_name = _required_setting("DEMO_API_FILES_BUCKET", settings.BUCKET)
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=_required_setting(
            "DEMO_API_FILES_AWS_ACCESS_KEY_ID", settings.AWS_ACCESS_KEY_ID
        ),
        aws_secret_access_key=_required_setting(
            "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY", settings.AWS_SECRET_ACCESS_KEY
        ),
        region_name=_required_setting(
            "DEMO_API_FILES_AWS_DEFAULT_REGION", settings.AWS_DEFAULT_REGION
        ),
    )
    bucket = S3Bucket(s3_client, bucket_name)
    extractions = Extractions(
        bucket,
        timeout_seconds=settings.EXTRACTION_TIMEOUT_SECONDS,
        threads=settings.EXTRACTION_THREADS,
        max_concurrent=settings.MAX_CONCURRENT_EXTRACTIONS,
    )

    app = FastAPI(
        title="LGOS Files API",
        version="0.1.0",
        lifespan=lifespan,
        # FastAPI records requests through the global providers; export belongs
        # to `opentelemetry-instrument`, so FastAPI must not add its own.
        telemetry={
            "auto_configure": False,
            "exclude": lambda scope: scope["path"].endswith("/health"),
        },
    )
    app.state.s3_client = s3_client
    app.state.extractions = extractions
    app.state.files = FileService(
        bucket, extractions, max_upload_bytes=settings.MAX_UPLOAD_BYTES
    )
    configure_openai_error_handlers(app)
    app.include_router(files_router)
    app.include_router(health_router)
    return app


def main() -> None:
    """Run the Files service."""
    import uvicorn

    uvicorn.run(
        "lgos_files_api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=settings.PORT,
    )


if __name__ == "__main__":
    main()
