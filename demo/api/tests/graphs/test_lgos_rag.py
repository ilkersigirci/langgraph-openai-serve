import io
import json
import re
import tarfile
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import httpx2
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve import GraphConfig, GraphError, GraphRegistry
from langgraph_openai_serve.graph.citations import citations_from_message
from langgraph_openai_serve.graph.events import status_description
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs import lgos_rag
from lgos_demo_api.graphs.lgos_rag import (
    DocsIndex,
    LgosRagSettings,
    Scope,
    SearchPlan,
    create_lgos_rag_graph,
    docs_release,
    fetch_chunks,
    load_chunks,
)

BASE_URL = "https://docs.example.com/lgos/1.0.0/"
ARCHIVE_URL = "https://github.example.com/lgos/archive/refs/tags/v1.0.0.tar.gz"
INSTALL_URL = f"{BASE_URL}getting-started/#install"
EXPIRY_URL = f"{BASE_URL}how-to-guides/server/#expire-paused-runs"
IDENTIFIER = "LGOS_INTERRUPT_TTL_MINUTES"
INSTALL = """\
| Python | Support |
| --- | --- |
| 3.11 | yes |

```bash
uv add langgraph-openai-serve
# a comment, not a heading
```"""
DEMO_URL = f"{BASE_URL}demo/docker/#start"
PAGES = {
    "demo/docker.md": "# Docker Compose\n\n## Start\n\nRun `just demo/compose`.\n",
    "getting-started.md": (
        f"# Get Started\n\nBuild an application.\n\n## Install\n\n{INSTALL}\n"
    ),
    "how-to-guides/index.md": (
        "# How-To Guides\n\n## Show `display_file`\n\nFirst.\n\n"
        "## Show `display_file`\n\nSecond.\n"
    ),
    "how-to-guides/server.md": (
        "# Run The Server\n\n## Expire Paused Runs\n\n"
        f"`{IDENTIFIER}` deletes runs left paused longer than this.\n"
        "See [settings](../reference.md#settings), [this section](#expire-paused-runs), "
        "and ![a diagram](../static/runs.png).\n"
    ),
    "reference.md": "# Reference\n",
}


class _StreamingToolModel(GenericFakeChatModel):
    """Stream reply text token by token, then its tool calls, like a provider."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_StreamingToolModel":  # ty: ignore[invalid-method-override]
        return self

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reply = next(self.messages)
        assert isinstance(reply, AIMessage)
        for token in filter(None, re.split(r"(\s)", reply.text)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": position,
                        "type": "tool_call_chunk",
                    }
                    for position, call in enumerate(reply.tool_calls)
                ],
                chunk_position="last",
            )
        )


async def _index() -> DocsIndex:
    return await DocsIndex.build(
        load_chunks(PAGES, BASE_URL),
        DeterministicFakeEmbedding(size=32),
    )


def _returning(index: DocsIndex) -> Callable[[], Awaitable[DocsIndex]]:
    async def docs_index() -> DocsIndex:
        return index

    return docs_index


def test_markdown_sections_become_chunks_linked_to_the_docs_site() -> None:
    chunks = load_chunks(PAGES, BASE_URL)

    assert [(chunk.metadata["title"], chunk.metadata["url"]) for chunk in chunks] == [
        ("Docker Compose > Start", DEMO_URL),
        ("Get Started", f"{BASE_URL}getting-started/"),
        ("Get Started > Install", INSTALL_URL),
        # Repeated headings number their anchors as the docs site does.
        (
            "How-To Guides > Show `display_file`",
            f"{BASE_URL}how-to-guides/#show-display_file",
        ),
        (
            "How-To Guides > Show `display_file`",
            f"{BASE_URL}how-to-guides/#show-display_file_1",
        ),
        ("Run The Server > Expire Paused Runs", EXPIRY_URL),
    ]
    assert chunks[2].page_content == f"Get Started > Install\n\n{INSTALL}"
    # Relative links point at the docs site, so a copied link still works.
    assert chunks[-1].page_content.endswith(
        f"See [settings]({BASE_URL}reference/#settings), "
        f"[this section]({EXPIRY_URL}), "
        f"and ![a diagram]({BASE_URL}static/runs.png)."
    )


def _archive(files: dict[str, str]) -> bytes:
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            member = tarfile.TarInfo(f"lgos-1.0.0/{name}")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return content.getvalue()


async def test_release_archive_supplies_only_its_docs_pages() -> None:
    archive = _archive(
        {
            "docs/how-to-guides/server.md": PAGES["how-to-guides/server.md"],
            "README.md": "# LGOS\n\nNot part of the documentation.\n",
        }
    )
    transport = httpx2.MockTransport(
        lambda _request: httpx2.Response(200, content=archive)
    )

    async with httpx2.AsyncClient(transport=transport) as client:
        chunks = await fetch_chunks(client, ARCHIVE_URL, BASE_URL)

    assert [chunk.metadata["url"] for chunk in chunks] == [EXPIRY_URL]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx2.Response(404), id="no-tag"),
        # The first LGOS tags predate the docs.
        pytest.param(
            httpx2.Response(200, content=_archive({"README.md": "# LGOS\n"})),
            id="tag-without-docs",
        ),
    ],
)
async def test_version_without_released_docs_is_a_graph_error(
    response: httpx2.Response,
) -> None:
    transport = httpx2.MockTransport(lambda _request: response)

    async with httpx2.AsyncClient(transport=transport) as client:
        with pytest.raises(GraphError, match="tagged LGOS release"):
            await fetch_chunks(client, ARCHIVE_URL, BASE_URL)


@pytest.mark.parametrize("bm25", [False, True])
@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("all", {"demo", "library"}),
        ("library", {"library"}),
        ("demo", {"demo"}),
    ],
)
async def test_scope_limits_results_to_its_docs(
    bm25: bool,
    scope: Scope,
    expected: set[str],
) -> None:
    index = await _index()

    # The query matches both parts, so only the scope can exclude one.
    results = await index.search(
        "Run LGOS with just demo/compose", bm25=bm25, scope=scope
    )

    assert {
        "demo" if result.metadata["url"].startswith(f"{BASE_URL}demo/") else "library"
        for result in results
    } == expected


DOCS_VERSIONS = [
    {"version": "2.1.0-rc1", "title": "2.1.0-rc1", "aliases": []},
    {"version": "2.0.0", "title": "2.0.0", "aliases": ["latest"]},
    {"version": "1.9.0", "title": "1.9.0", "aliases": []},
]


def _docs_site() -> httpx2.MockTransport:
    return httpx2.MockTransport(
        lambda _request: httpx2.Response(200, json=DOCS_VERSIONS)
    )


@pytest.mark.parametrize(
    ("installed", "release"),
    [
        pytest.param("0.0.1", "2.0.0", id="checkout-uses-latest"),
        pytest.param("1.9.0", "1.9.0", id="release-uses-itself"),
        # Installed versions are normalized, unlike the tag and the docs site.
        pytest.param("2.1.0rc1", "2.1.0-rc1", id="prerelease-uses-its-tag-name"),
    ],
)
async def test_docs_release_follows_the_installed_lgos(
    monkeypatch: pytest.MonkeyPatch,
    installed: str,
    release: str,
) -> None:
    monkeypatch.setattr(lgos_rag, "version", lambda _name: installed)

    async with httpx2.AsyncClient(transport=_docs_site()) as client:
        assert await docs_release(client) == release


async def test_release_without_published_docs_is_a_graph_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lgos_rag, "version", lambda _name: "2.2.0")

    async with httpx2.AsyncClient(transport=_docs_site()) as client:
        with pytest.raises(GraphError, match=r"no docs for LGOS 2\.2\.0"):
            await docs_release(client)


class _IdentifierBlindEmbeddings(Embeddings):
    """Rank the section naming the identifier last, as weak embeddings can."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if IDENTIFIER in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


async def test_bm25_puts_an_exact_identifier_match_first() -> None:
    index = await DocsIndex.build(
        load_chunks(PAGES, BASE_URL), _IdentifierBlindEmbeddings()
    )

    semantic = await index.search(IDENTIFIER, bm25=False)
    fused = await index.search(IDENTIFIER, bm25=True)

    assert semantic[-1].metadata["url"] == EXPIRY_URL
    assert fused[0].metadata["url"] == EXPIRY_URL


def _plan(*queries: str, needs_docs: bool = True, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": SearchPlan.__name__,
                "args": {"needs_docs": needs_docs, "queries": list(queries)},
                "id": "plan-1",
                "type": "tool_call",
            }
        ],
    )


async def _stream(
    make_graph_input: Callable[..., Any],
    graph: CompiledStateGraph,
    question: str,
) -> tuple[list[object], AIMessage]:
    registry = GraphRegistry(
        graphs={
            "lgos-rag": GraphConfig(
                graph=graph,
                description="DUMMY",
                client_settings=LgosRagSettings,
            )
        }
    )
    graph_request, messages = make_graph_input("lgos-rag", content=question)
    events = [
        event async for event in run_langgraph_stream(graph_request, messages, registry)
    ]
    answer = events[-1]
    assert isinstance(answer, AIMessage)
    timeline = [
        status_description(event["data"]) if isinstance(event, dict) else event
        for event in events[:-1]
    ]
    return timeline, answer


async def test_only_the_cited_answer_streams_after_planned_searches(
    make_graph_input,
) -> None:
    reply = (
        f"Run `uv add langgraph-openai-serve` as [Install]({INSTALL_URL}) shows, "
        "not as [another site](https://example.com/other) says."
    )
    model = _StreamingToolModel(
        messages=iter(
            [
                # The plan's own text stays private, like every planning token.
                _plan("install LGOS", content="Planning the search. "),
                AIMessage(content=reply),
            ]
        )
    )
    graph = create_lgos_rag_graph(model, docs_index=_returning(await _index()))

    timeline, answer = await _stream(make_graph_input, graph, "How do I install LGOS?")

    assert timeline[:2] == [
        "Understanding the question",
        "Searching the docs: install LGOS",
    ]
    assert "".join(str(event) for event in timeline[2:]) == answer.text == reply
    assert [
        (citation["url"], citation["title"])
        for citation in citations_from_message(answer)
    ] == [(INSTALL_URL, "Get Started > Install")]


async def test_messages_needing_no_docs_skip_the_search(make_graph_input) -> None:
    async def unused_index() -> DocsIndex:
        pytest.fail("A message that needs no docs must not load the index.")

    model = _StreamingToolModel(
        messages=iter(
            [
                _plan("capital of France", needs_docs=False),
                AIMessage(content="I only answer questions about LGOS."),
            ]
        )
    )
    graph = create_lgos_rag_graph(model, docs_index=unused_index)

    timeline, answer = await _stream(
        make_graph_input, graph, "What is the capital of France?"
    )

    assert timeline[0] == "Understanding the question"
    assert "".join(str(event) for event in timeline[1:]) == answer.text
    assert answer.text == "I only answer questions about LGOS."
