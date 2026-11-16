"""
Content extraction, run by the API process in the background.

Its state lives in S3 next to the content, never only in process memory:

* "content/{file_id}.json": the extracted content; the file is "processed".
* "extraction/{file_id}.error": the extraction failed for good ("error"); the
  body holds its status code and message.
* "extraction/{file_id}.attempts": how many extractions of the file got a slot
  without ending, such as one a shutdown cut short.

A file to extract with neither content nor an error reads as "uploaded": this
process is extracting it, or its next retrieve starts the extraction again, as
after a restart. Only extractions that get a slot count as attempts, so restarts
never fail a file that was only waiting for one. After MAX_ATTEMPTS unfinished
attempts the file fails, so a document that stops the service, for example by
running it out of memory, does not stop it again on every retrieve. Failures of
the file itself are stored; others, such as of storage, are not, so the next
retrieve tries again within the same attempts. All of this assumes one API
process per bucket.
"""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING

import anyio
from fastapi import status

from lgos_files_api.api.files import native, xberg
from lgos_files_api.api.files.schemas import FileStatus
from lgos_files_api.storage import S3Bucket

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = logging.getLogger(__name__)

CONTENT_PREFIX, MARKER_PREFIX = "content/", "extraction/"
ATTEMPTS, ERROR = "attempts", "error"
MAX_ATTEMPTS = 2
# Open WebUI's OPENAI_FILES_API_FILE_EXTENSIONS lists them too and sends them to
# graphs by ID, which forward the originals to OpenAI: each is a type OpenAI's
# file inputs accept.
EXTRACTED_SUFFIXES = native.SUFFIXES | xberg.SUFFIXES


@dataclass(frozen=True, slots=True)
class Extraction:
    status: FileStatus
    details: str | None = None
    # Of a failed extraction:
    status_code: int | None = None


UPLOADED = Extraction("uploaded")
PROCESSED = Extraction("processed")


def is_supported(filename: str) -> bool:
    """Return whether content is extracted for the file."""
    return PurePath(filename).suffix.lower() in EXTRACTED_SUFFIXES


def content_key(file_id: str) -> str:
    return f"{CONTENT_PREFIX}{file_id}.json"


def _marker_key(file_id: str, kind: str) -> str:
    return f"{MARKER_PREFIX}{file_id}.{kind}"


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
        """Run extractions for the app's lifetime."""
        async with anyio.create_task_group() as group:
            self._group = group
            try:
                yield
            finally:
                # An unfinished extraction starts again on its file's next retrieve.
                self._group = None
                group.cancel_scope.cancel()

    def start(self, file_id: str, filename: str) -> None:
        """Extract the file in the background, unless this process already is."""
        if file_id in self._running:
            return
        if self._group is None:
            msg = "Extractions.run() is not running."
            raise RuntimeError(msg)
        self._running.add(file_id)
        self._group.start_soon(self._run, file_id, filename)

    async def read(self, file_id: str) -> Extraction | None:
        """Return the file's extraction, or None if none runs here and none ended."""
        # In the reverse order of an ending extraction, which stores its outcome
        # before it stops running, so this never misses both.
        if file_id in self._running:
            return UPLOADED
        if await self._bucket.exists(content_key(file_id)):
            return PROCESSED
        return await self._failure(file_id)

    async def read_all(self) -> dict[str, Extraction]:
        """Return the ended extractions of every file, from two listings."""
        states = {
            stored.key.removeprefix(CONTENT_PREFIX).removesuffix(".json"): PROCESSED
            for stored in await self._bucket.list_objects(CONTENT_PREFIX)
        }
        for marker in await self._bucket.list_objects(MARKER_PREFIX):
            file_id, kind = marker.key.removeprefix(MARKER_PREFIX).rsplit(".", 1)
            if kind != ERROR:
                continue
            failure = await self._failure(file_id)
            if failure is not None:  # None: deleted since the listing
                states[file_id] = failure
        return states

    async def forget(self, file_id: str) -> None:
        """Remove the content and the markers of a deleted file."""
        for key in (
            content_key(file_id),
            _marker_key(file_id, ERROR),
            _marker_key(file_id, ATTEMPTS),
        ):
            await self._bucket.delete(key)

    async def _failure(self, file_id: str) -> Extraction | None:
        stored = await self._bucket.get(_marker_key(file_id, ERROR))
        if stored is None:
            return None
        body = json.loads(await stored.read())
        return Extraction("error", body["message"], body["status_code"])

    async def _run(self, file_id: str, filename: str) -> None:
        """Extract one file once it gets a slot; never raises."""
        try:
            # The slot bounds the documents held in memory, not only the extractors.
            async with self._slots:
                await self._extract(file_id, filename)
        except Exception:
            # Raised, it would cancel the task group and with it the app. Not
            # stored: the file's next retrieve tries again, as another attempt.
            logger.exception("Extracting %s failed", file_id)
        finally:
            # Last, so `read` sees the extraction running until its outcome is stored.
            self._running.discard(file_id)

    async def _extract(self, file_id: str, filename: str) -> None:
        """Count the attempt, then store the content or the file's failure."""
        attempts = await self._attempts(file_id)
        if attempts >= MAX_ATTEMPTS:
            logger.warning(
                "%s did not finish extracting in %d attempts", file_id, attempts
            )
            await self._fail(
                file_id,
                filename,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "The extraction did not finish.",
            )
            return
        await self._bucket.put(
            _marker_key(file_id, ATTEMPTS),
            json.dumps({"attempts": attempts + 1}).encode(),
            content_type="application/json",
        )
        original = await self._bucket.get(f"{file_id}/{filename}")
        if original is None:  # deleted while waiting for the slot
            await self.forget(file_id)
            return
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
            await self._fail(
                file_id,
                filename,
                status.HTTP_504_GATEWAY_TIMEOUT,
                "Timed out while parsing the file.",
            )
            return
        except ValueError as error:
            logger.warning("Failed to extract %s: %s", file_id, error)
            await self._fail(
                file_id,
                filename,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "Failed to parse the file.",
            )
            return
        await self._end(file_id, filename, content_key(file_id), content)

    async def _attempts(self, file_id: str) -> int:
        stored = await self._bucket.get(_marker_key(file_id, ATTEMPTS))
        return json.loads(await stored.read())["attempts"] if stored else 0

    async def _fail(
        self, file_id: str, filename: str, status_code: int, message: str
    ) -> None:
        body = {"status_code": status_code, "message": message}
        await self._end(
            file_id, filename, _marker_key(file_id, ERROR), json.dumps(body).encode()
        )

    async def _end(self, file_id: str, filename: str, key: str, outcome: bytes) -> None:
        """Store the extraction's outcome, which ends its attempts."""
        await self._bucket.put(key, outcome, content_type="application/json")
        await self._bucket.delete(_marker_key(file_id, ATTEMPTS))
        # Deleted while extracting: remove the outcome instead of orphaning it.
        if not await self._bucket.exists(f"{file_id}/{filename}"):
            await self.forget(file_id)
