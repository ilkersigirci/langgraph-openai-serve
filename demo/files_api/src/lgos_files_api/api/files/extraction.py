"""
Content extraction, run by the API process in the background (`Extractions.run`).

Its state lives in S3 next to the content, never only in process memory:

* "content/{file_id}.json": the extracted content; the file is "processed".
* "extraction/{file_id}.queued": an extraction is queued or running ("uploaded");
  the body counts the times a start of the API resumed it.
* "extraction/{file_id}.error": the last extraction failed ("error"); the body
  holds its status code and message.

The API starts extractions (`start`) and reads their state (`read`, `read_all`).
Extractions cut short by a shutdown keep their queued marker and start again
when the API starts, once: one that a second start still finds unfinished
fails, so a document that stops the API, for example by running out of memory,
cannot stop every later start. A queued marker whose extraction this process is
not running was lost, for example to a failed S3 write, and reads as no state,
so the next retrieve starts it again. Both assume one API process per bucket.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
from fastapi import status

from lgos_files_api.api.files import native, xberg
from lgos_files_api.api.files.schemas import FileStatus
from lgos_files_api.storage import S3Bucket, StoredObject

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = logging.getLogger(__name__)

QUEUED, ERROR = "queued", "error"
CONTENT_PREFIX, MARKER_PREFIX = "content/", "extraction/"
MAX_RESUMES = 1


@dataclass(frozen=True, slots=True)
class Extraction:
    status: FileStatus
    details: str | None = None
    # Of a failed extraction:
    status_code: int | None = None
    failed_at: float | None = None


QUEUED_STATE = Extraction("uploaded")
PROCESSED_STATE = Extraction("processed")


def is_supported(filename: str) -> bool:
    """Return whether content is extracted for the file."""
    return native.is_supported(filename) or xberg.is_supported(filename)


def content_key(file_id: str) -> str:
    return f"{CONTENT_PREFIX}{file_id}.json"


def _marker_key(file_id: str, kind: str) -> str:
    return f"{MARKER_PREFIX}{file_id}.{kind}"


def _marker(stored: StoredObject) -> tuple[str, str]:
    """Return the file ID and kind of a marker."""
    file_id, kind = stored.key.removeprefix(MARKER_PREFIX).rsplit(".", 1)
    return file_id, kind


class Extractions:
    """The extractions of one bucket's files, at most `max_concurrent` at once."""

    def __init__(
        self,
        bucket: S3Bucket,
        *,
        timeout_seconds: int,
        threads: int,
        max_concurrent: int,
    ) -> None:
        self._bucket = bucket
        self._timeout_seconds = timeout_seconds
        self._threads = threads
        self._slots = anyio.CapacityLimiter(max_concurrent)
        self._group: TaskGroup | None = None
        # One extraction per file at a time.
        self._running: set[str] = set()

    @asynccontextmanager
    async def run(self) -> AsyncGenerator[None]:
        """Run extractions for the app's lifetime; resume those a shutdown cut short."""
        async with anyio.create_task_group() as group:
            self._group = group
            for marker in await self._bucket.list_objects(MARKER_PREFIX):
                file_id, kind = _marker(marker)
                if kind == QUEUED and (
                    originals := await self._bucket.list_objects(f"{file_id}/")
                ):
                    await self._resume(file_id, originals[0].key.split("/", 1)[1])
            try:
                yield
            finally:
                # Unfinished extractions keep their queued marker: the next
                # start resumes them.
                self._group = None
                group.cancel_scope.cancel()

    async def start(self, file_id: str, filename: str) -> None:
        """Extract the file in the background; it reads "uploaded" until then."""
        await self._put_queued(file_id, resumes=0)
        await self._bucket.delete(_marker_key(file_id, ERROR))
        self._schedule(file_id, filename)

    async def read(self, file_id: str) -> Extraction | None:
        """Return the file's extraction: None without content, a run, or an error."""
        # In the reverse order of an ending extraction, which stores the
        # content before it stops running, so this never misses both.
        if file_id in self._running:
            return QUEUED_STATE
        if await self._bucket.exists(content_key(file_id)):
            return PROCESSED_STATE
        kinds = {
            _marker(marker)[1]
            for marker in await self._bucket.list_objects(f"{MARKER_PREFIX}{file_id}.")
        }
        return await self._from_markers(file_id, kinds)

    async def read_all(self) -> dict[str, Extraction]:
        """Return the extractions of every file that has one, from two listings."""
        states = {
            stored.key.removeprefix(CONTENT_PREFIX).removesuffix(".json"): (
                PROCESSED_STATE
            )
            for stored in await self._bucket.list_objects(CONTENT_PREFIX)
        }
        markers: dict[str, set[str]] = {}
        for marker in await self._bucket.list_objects(MARKER_PREFIX):
            file_id, kind = _marker(marker)
            markers.setdefault(file_id, set()).add(kind)
        for file_id, kinds in markers.items():
            if file_id in states:
                continue
            state = await self._from_markers(file_id, kinds)
            if state is None and QUEUED in kinds:
                # Lost; the file's next retrieve starts it again.
                state = QUEUED_STATE
            if state is not None:
                states[file_id] = state
        return states

    async def forget(self, file_id: str) -> None:
        """Remove the content and the markers of a deleted file."""
        for key in (
            content_key(file_id),
            _marker_key(file_id, QUEUED),
            _marker_key(file_id, ERROR),
        ):
            await self._bucket.delete(key)

    async def _from_markers(self, file_id: str, kinds: set[str]) -> Extraction | None:
        """
        Return the state of a file without content, from its marker kinds.

        A running extraction wins over the error it retries, and over a queued
        marker it deleted on finishing, which would otherwise read as lost.
        """
        if file_id in self._running:
            return QUEUED_STATE
        if ERROR not in kinds:
            return None
        error = await self._bucket.get(_marker_key(file_id, ERROR))
        if error is None:  # replaced by a retry since the listing
            return None
        body = json.loads(await error.read())
        return Extraction(
            "error", body["message"], body["status_code"], body["failed_at"]
        )

    async def _resume(self, file_id: str, filename: str) -> None:
        queued = await self._bucket.get(_marker_key(file_id, QUEUED))
        resumes = json.loads(await queued.read()).get("resumes", 0) if queued else 0
        if resumes >= MAX_RESUMES:
            logger.warning("The extraction of %s was cut short twice", file_id)
            await self._record_failure(
                file_id,
                filename,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "The extraction did not finish.",
            )
            return
        logger.info("Resuming the extraction of %s", file_id)
        await self._put_queued(file_id, resumes=resumes + 1)
        self._schedule(file_id, filename)

    async def _put_queued(self, file_id: str, *, resumes: int) -> None:
        await self._bucket.put(
            _marker_key(file_id, QUEUED),
            json.dumps({"resumes": resumes}).encode(),
            content_type="application/json",
        )

    def _schedule(self, file_id: str, filename: str) -> None:
        if file_id in self._running:
            return
        if self._group is None:
            msg = "Extractions.run() is not running."
            raise RuntimeError(msg)
        self._running.add(file_id)
        self._group.start_soon(self._run, file_id, filename)

    async def _run(self, file_id: str, filename: str) -> None:
        """Extract one file; its failure is stored, never raised."""
        try:
            failure = await self._failure(file_id, filename)
            if failure is not None:
                await self._record_failure(file_id, filename, *failure)
        except Exception:
            # Raised, it would cancel the task group and with it the app. The
            # queued marker stays, so the next retrieve starts the file again.
            logger.exception("Storing the failed extraction of %s failed", file_id)
        finally:
            # Last, so `read` sees the extraction running until its outcome is stored.
            self._running.discard(file_id)

    async def _failure(self, file_id: str, filename: str) -> tuple[int, str] | None:
        """Extract the file; return the status code and message of its failure."""
        try:
            return await self._extract(file_id, filename)
        except Exception:
            # Recorded, unlike a lost marker, so that polling clients see the
            # error instead of starting the extraction again on every check.
            logger.exception("Extracting %s failed", file_id)
            return status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal Server Error"

    async def _extract(self, file_id: str, filename: str) -> tuple[int, str] | None:
        """Extract and store the content; return the failure of an unreadable file."""
        key = f"{file_id}/{filename}"
        # The slot bounds the documents held in memory, not only the extractors.
        async with self._slots:
            original = await self._bucket.get(key)
            if original is None:  # deleted, which removed its markers too
                return None
            document = await original.read()
            try:
                content = (
                    await anyio.to_thread.run_sync(native.extract, filename, document)
                    if native.is_supported(filename)
                    else await xberg.extract(
                        filename,
                        document,
                        timeout_seconds=self._timeout_seconds,
                        threads=self._threads,
                    )
                )
            except TimeoutError as error:
                logger.warning("Timed out extracting %s: %s", file_id, error)
                return (
                    status.HTTP_504_GATEWAY_TIMEOUT,
                    "Timed out while parsing the file.",
                )
            except ValueError as error:
                logger.warning("Failed to extract %s: %s", file_id, error)
                return (
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "Failed to parse the file.",
                )
        await self._bucket.put(
            content_key(file_id), content, content_type="application/json"
        )
        await self._bucket.delete(_marker_key(file_id, QUEUED))
        # Deleted while extracting: remove the content instead of orphaning it.
        if not await self._bucket.exists(key):
            await self.forget(file_id)
        return None

    async def _record_failure(
        self, file_id: str, filename: str, status_code: int, message: str
    ) -> None:
        """Store the error marker, unless the file was deleted meanwhile."""
        if not await self._bucket.exists(f"{file_id}/{filename}"):
            return
        body = {
            "status_code": status_code,
            "message": message,
            "failed_at": time.time(),
        }
        await self._bucket.put(
            _marker_key(file_id, ERROR),
            json.dumps(body).encode(),
            content_type="application/json",
        )
        await self._bucket.delete(_marker_key(file_id, QUEUED))
        # Deleted since the check: remove the marker instead of orphaning it.
        if not await self._bucket.exists(f"{file_id}/{filename}"):
            await self.forget(file_id)
