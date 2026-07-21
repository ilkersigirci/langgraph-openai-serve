from collections import deque
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs import lgos_rag as lgos_rag_module

ANSWER = (
    "Use the [registered model](https://example.com/second) through the OpenAI "
    "client. View the ![architecture diagram](https://example.com/diagram.png) "
    "or follow the [audio link](https://example.com/overview.mp3)."
)
DECISION_PREAMBLE = "I will check the documentation. "
HISTORY_ANSWER = 'You asked: "Who are you?"'
REWRITTEN_QUESTION = "How do I call LGOS using an OpenAI client?"


def _tool_call(
    query: str,
    call_id: str = "retrieve-1",
    content: str = "",
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "retrieve_lgos_rag",
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _stub_runnable(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    *responses: Any,
) -> None:
    remaining = deque(responses)

    async def respond(_: Any) -> Any:
        if not remaining:
            raise AssertionError(f"Unexpected call to {factory_name}")
        return remaining.popleft()

    runnable = RunnableLambda(respond)
    monkeypatch.setattr(lgos_rag_module, factory_name, lambda: runnable)


def _stub_chat_model(
    monkeypatch: pytest.MonkeyPatch,
    *responses: str,
) -> None:
    model = FakeListChatModel(responses=list(responses))
    monkeypatch.setattr(lgos_rag_module, "_chat_model", lambda: model)


def _registry() -> GraphRegistry:
    return GraphRegistry(
        registry={
            "lgos-rag": GraphConfig(
                graph=lgos_rag_module.lgos_rag,
                streamable_node_names=[
                    "generate_query_or_respond",
                    "generate_answer",
                    "answer_no_results",
                ],
            )
        }
    )


def _source_document(content: str, name: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"docs/{name.lower()}.md",
            "title": name,
            "url": f"https://example.com/{name.lower()}",
        },
    )


async def _stream_text(request: ChatCompletionRequest) -> str:
    events = [
        event
        async for event in run_langgraph_stream(
            request.model,
            request.messages,
            _registry(),
            request,
        )
    ]
    assert all(isinstance(event, str) for event in events)
    return "".join(event for event in events if isinstance(event, str))


def test_formats_context_with_source_metadata_and_markdown() -> None:
    markdown = (
        "Read the [guide](https://example.com/guide) and view "
        "![Diagram](https://example.com/diagram.png)."
    )

    context = lgos_rag_module._format_context([_source_document(markdown, "Reference")])

    assert context == (
        f"Title: Reference\nSource URL: https://example.com/reference\n{markdown}"
    )


def test_loads_a_controlled_markdown_corpus(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    nested = docs_root / "nested"
    nested.mkdir(parents=True)
    (docs_root / "guide.md").write_text("# Guide\n\nBody.", encoding="utf-8")
    (nested / "reference.md").write_text("No heading.", encoding="utf-8")

    documents = lgos_rag_module.load_documents(
        docs_root,
        "https://example.com/lgos-docs",
    )

    assert [(document.page_content, document.metadata) for document in documents] == [
        (
            "# Guide\n\nBody.",
            {
                "source": "guide.md",
                "title": "Guide (guide.md)",
                "url": "https://example.com/lgos-docs/guide.md",
            },
        ),
        (
            "No heading.",
            {
                "source": "nested/reference.md",
                "title": "nested/reference.md (nested/reference.md)",
                "url": "https://example.com/lgos-docs/nested/reference.md",
            },
        ),
    ]


def test_missing_corpus_reports_the_missing_markdown(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No Markdown documentation"):
        lgos_rag_module.load_documents(tmp_path / "missing")


def test_default_corpus_is_bundled_with_the_api_package() -> None:
    expected_sources = {
        path.relative_to(lgos_rag_module.DOCS_ROOT).as_posix()
        for path in lgos_rag_module.DOCS_ROOT.rglob("*.md")
    }

    documents = lgos_rag_module.load_documents()

    assert expected_sources == {
        "demo-models.md",
        "openai-clients.md",
        "overview.md",
    }
    assert {document.metadata["source"] for document in documents} == expected_sources


def test_splits_documents_and_preserves_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_size = 60
    monkeypatch.setattr(lgos_rag_module, "CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(lgos_rag_module, "CHUNK_OVERLAP", 10)
    document = _source_document("alpha beta gamma delta " * 12, "Reference")

    chunks = lgos_rag_module.split_documents([document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= chunk_size for chunk in chunks)
    assert all(chunk.metadata["source"] == "docs/reference.md" for chunk in chunks)
    assert all(isinstance(chunk.metadata["start_index"], int) for chunk in chunks)


@pytest.mark.anyio
async def test_retrieval_uses_the_rewritten_query_and_streams_only_the_answer(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        _source_document("First source", "First"),
        _source_document(
            "Second source\n\n"
            "![Architecture diagram](https://example.com/diagram.png)\n\n"
            "[Audio overview](https://example.com/overview.mp3)",
            "Second",
        ),
    ]
    queries: list[str] = []

    async def retrieve_documents(query: str) -> list[Document]:
        queries.append(query)
        return documents

    monkeypatch.setattr(lgos_rag_module, "_retrieve_documents", retrieve_documents)
    _stub_runnable(
        monkeypatch,
        "_retrieval_decider",
        _tool_call(REWRITTEN_QUESTION, content=DECISION_PREAMBLE),
    )
    _stub_runnable(
        monkeypatch,
        "_document_grader",
        lgos_rag_module.GradeDocuments(binary_score="yes"),
    )
    _stub_chat_model(monkeypatch, ANSWER)
    request = make_request(
        "lgos-rag",
        messages=[
            {"role": "user", "content": "What is LGOS?"},
            {"role": "assistant", "content": "It serves LangGraph over /v1."},
            {"role": "user", "content": "How do I call it?"},
        ],
    )

    streamed_answer = await _stream_text(request)

    assert queries == [REWRITTEN_QUESTION]
    assert streamed_answer == ANSWER
    assert DECISION_PREAMBLE not in streamed_answer


@pytest.mark.anyio
async def test_direct_response_skips_retrieval(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_retrieval(query: str) -> list[Document]:
        raise AssertionError(f"Conversation unexpectedly retrieved: {query}")

    monkeypatch.setattr(lgos_rag_module, "_retrieve_documents", unexpected_retrieval)
    _stub_runnable(
        monkeypatch,
        "_retrieval_decider",
        AIMessage(content="respond_direct"),
    )
    _stub_chat_model(monkeypatch, HISTORY_ANSWER)
    request = make_request(
        "lgos-rag",
        messages=[
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am the LGOS RAG assistant."},
            {"role": "user", "content": "What did I ask you?"},
        ],
    )

    assert await _stream_text(request) == HISTORY_ANSWER


@pytest.mark.anyio
async def test_irrelevant_retrieval_rewrites_once_then_stops(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []

    async def retrieve_documents(query: str) -> list[Document]:
        queries.append(query)
        return [_source_document("Unrelated content", "Irrelevant")]

    monkeypatch.setattr(lgos_rag_module, "_retrieve_documents", retrieve_documents)
    _stub_runnable(
        monkeypatch,
        "_retrieval_decider",
        _tool_call("weak query", "retrieve-1"),
        _tool_call(REWRITTEN_QUESTION, "retrieve-2"),
    )
    _stub_runnable(
        monkeypatch,
        "_document_grader",
        lgos_rag_module.GradeDocuments(binary_score="no"),
        lgos_rag_module.GradeDocuments(binary_score="no"),
    )
    _stub_runnable(
        monkeypatch,
        "_internal_chat_model",
        AIMessage(content=REWRITTEN_QUESTION),
    )
    refusal = "I cannot answer that from the available LGOS documentation."
    _stub_chat_model(monkeypatch, refusal)
    request = make_request("lgos-rag", content="How do I call it?")

    streamed_answer = await _stream_text(request)

    assert queries == ["weak query", REWRITTEN_QUESTION]
    assert streamed_answer == refusal
