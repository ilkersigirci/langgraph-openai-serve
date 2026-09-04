"""Standalone OpenAI-compatible Files service backed by S3."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI
from starlette.types import Lifespan

from lgos_files_api.api import configure_openai_error_handlers, router
from lgos_files_api.contracts import FileRepository
from lgos_files_api.s3 import S3FileRepository
from lgos_files_api.settings import settings


def _required_setting(name: str, value: str | None) -> str:
    if value:
        return value
    msg = f"{name} is required by lgos-files-api."
    raise RuntimeError(msg)


def create_files_app(
    repository: FileRepository,
    *,
    lifespan: Lifespan[FastAPI] | None = None,
) -> FastAPI:
    """Create the Files application around one repository."""
    app = FastAPI(
        title="LGOS Files API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.file_repository = repository
    configure_openai_error_handlers(app)
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the synchronous boto3 client during shutdown."""
    try:
        yield
    finally:
        app.state.s3_client.close()


def create_s3_app() -> FastAPI:
    """Create the configured S3-backed Files service."""
    bucket = _required_setting("DEMO_API_FILES_BUCKET", settings.BUCKET)
    access_key_id = _required_setting(
        "DEMO_API_FILES_AWS_ACCESS_KEY_ID",
        settings.AWS_ACCESS_KEY_ID,
    )
    secret_access_key = _required_setting(
        "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY",
        settings.AWS_SECRET_ACCESS_KEY,
    )
    region = _required_setting(
        "DEMO_API_FILES_AWS_DEFAULT_REGION",
        settings.AWS_DEFAULT_REGION,
    )
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
    )
    app = create_files_app(
        S3FileRepository(s3_client, bucket=bucket),
        lifespan=lifespan,
    )
    app.state.s3_client = s3_client
    return app


def main() -> None:
    """Run the Files service."""
    import uvicorn

    uvicorn.run(
        "lgos_files_api.app:create_s3_app",
        factory=True,
        host="0.0.0.0",
        port=settings.PORT,
    )


if __name__ == "__main__":
    main()
