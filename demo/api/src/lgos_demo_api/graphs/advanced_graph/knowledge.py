"""Small adapter for an OpenAI-compatible Files and vector-store service."""

from dataclasses import dataclass
from typing import Literal, Protocol

from anyio import fail_after
from openai import APIError, AsyncOpenAI, NotFoundError

IndexStatus = Literal["uploaded", "indexed", "index_failed"]


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    file_id: str
    filename: str
    text: str


class KnowledgeBase(Protocol):
    """Operations the graph needs from any compatible vector-store service."""

    @property
    def vector_store_id(self) -> str: ...

    async def search(self, query: str) -> list[KnowledgeResult]: ...

    async def upload(self, filename: str, content: bytes) -> str: ...

    async def index(self, file_id: str) -> IndexStatus: ...


@dataclass(frozen=True, slots=True)
class OpenAICompatibleKnowledgeBase:
    """Use only standard OpenAI Files and vector-store endpoints."""

    client: AsyncOpenAI
    vector_store_id: str

    async def search(self, query: str) -> list[KnowledgeResult]:
        page = await self.client.vector_stores.search(
            self.vector_store_id,
            query=query,
            max_num_results=5,
        )
        return [
            KnowledgeResult(
                file_id=result.file_id,
                filename=result.filename,
                text="\n".join(part.text for part in result.content)[:4_000],
            )
            for result in page.data
            if result.file_id and result.filename and result.content
        ]

    async def upload(self, filename: str, content: bytes) -> str:
        uploaded = await self.client.files.create(
            file=(filename, content, "text/markdown"),
            purpose="user_data",
        )
        return uploaded.id

    async def index(self, file_id: str) -> IndexStatus:
        try:
            with fail_after(60):
                try:
                    await self.client.vector_stores.files.retrieve(
                        file_id,
                        vector_store_id=self.vector_store_id,
                    )
                except NotFoundError:
                    await self.client.vector_stores.files.create(
                        vector_store_id=self.vector_store_id,
                        file_id=file_id,
                    )
                result = await self.client.vector_stores.files.poll(
                    file_id,
                    vector_store_id=self.vector_store_id,
                    poll_interval_ms=1_000,
                )
        except (APIError, TimeoutError):
            return "uploaded"
        return "indexed" if result.status == "completed" else "index_failed"


__all__ = [
    "IndexStatus",
    "KnowledgeBase",
    "KnowledgeResult",
    "OpenAICompatibleKnowledgeBase",
]
