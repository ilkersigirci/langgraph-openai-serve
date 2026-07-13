from pathlib import Path
from typing import Literal

import pytest
from demo.api.graphs import lgos_rag as lgos_rag_module
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import CustomStreamPart
from openai.types.chat.chat_completion_message import Annotation

from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import (
    LangGraphInterrupt,
    run_langgraph_stream,
)

ANSWER = "Use the registered model through the OpenAI client. [2]"
DECISION_PREAMBLE = "I will check the documentation. "
HISTORY_ANSWER = 'You asked: "Who are you?"'
REWRITTEN_QUESTION = "How do I call LGOS using an OpenAI client?"


def tool_call(
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


def stub_decisions(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[AIMessage],
) -> None:
    remaining = iter(responses)

    async def respond(_: list[BaseMessage]) -> AIMessage:
        return next(remaining)

    monkeypatch.setattr(
        lgos_rag_module,
        "_retrieval_decider",
        lambda: RunnableLambda(respond),
    )


def stub_grades(
    monkeypatch: pytest.MonkeyPatch,
    grades: list[Literal["yes", "no"]],
) -> None:
    remaining = iter(grades)

    async def respond(_: list[BaseMessage]) -> lgos_rag_module.GradeDocuments:
        return lgos_rag_module.GradeDocuments(binary_score=next(remaining))

    monkeypatch.setattr(
        lgos_rag_module,
        "_document_grader",
        lambda: RunnableLambda(respond),
    )


def lgos_rag_registry() -> GraphRegistry:
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


def source_document(content: str, name: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"docs/{name.lower()}.md",
            "title": name,
            "url": f"https://example.com/{name.lower()}",
        },
    )


def test_lgos_rag_loads_and_splits_every_markdown_document() -> None:
    documents = lgos_rag_module.load_documents()
    expected_sources = {
        path.relative_to(lgos_rag_module.DOCS_ROOT.parent).as_posix()
        for path in lgos_rag_module.DOCS_ROOT.rglob("*.md")
    }

    assert {document.metadata["source"] for document in documents} == expected_sources
    for document in documents:
        source = str(document.metadata["source"])
        path = lgos_rag_module.DOCS_ROOT.parent / Path(source)
        assert document.page_content == path.read_text(encoding="utf-8")
        assert document.metadata["url"] == f"{lgos_rag_module.DOCS_BASE_URL}{source}"

    chunks = lgos_rag_module.split_documents(documents)
    assert len(chunks) > len(documents)
    assert {chunk.metadata["source"] for chunk in chunks} == expected_sources
    assert all("start_index" in chunk.metadata for chunk in chunks)
    assert (
        max(len(chunk.page_content) for chunk in chunks) <= lgos_rag_module.CHUNK_SIZE
    )


@pytest.mark.anyio
async def test_lgos_rag_retrieves_grades_streams_and_cites(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        source_document("First source", "First"),
        source_document("Second source", "Second"),
    ]
    queries: list[str] = []

    async def retrieve_documents(query: str) -> list[Document]:
        queries.append(query)
        return documents

    monkeypatch.setattr(lgos_rag_module, "_retrieve_documents", retrieve_documents)
    stub_decisions(
        monkeypatch,
        [tool_call(REWRITTEN_QUESTION, content=DECISION_PREAMBLE)],
    )
    stub_grades(monkeypatch, ["yes"])
    answer_model = FakeListChatModel(responses=[ANSWER])
    monkeypatch.setattr(lgos_rag_module, "_chat_model", lambda: answer_model)
    request = make_request(
        "lgos-rag",
        messages=[
            {"role": "user", "content": "What is LGOS?"},
            {"role": "assistant", "content": "It serves LangGraph over /v1."},
            {"role": "user", "content": "How do I call it?"},
        ],
    )

    events = [
        event
        async for event in run_langgraph_stream(
            "lgos-rag",
            request.messages,
            lgos_rag_registry(),
            request,
        )
    ]

    assert queries == [REWRITTEN_QUESTION]
    streamed_answer = "".join(event for event in events if isinstance(event, str))
    assert streamed_answer == ANSWER
    assert DECISION_PREAMBLE not in streamed_answer
    custom_events: list[CustomStreamPart] = [
        event for event in events if not isinstance(event, (str, LangGraphInterrupt))
    ]
    assert len(custom_events) == 1
    citation = Annotation.model_validate(custom_events[0]["data"]).url_citation
    assert citation.title == "Second"
    assert citation.url == "https://example.com/second"
    assert streamed_answer[citation.start_index : citation.end_index] == "[2]"


@pytest.mark.anyio
async def test_lgos_rag_answers_conversation_without_retrieval_or_citations(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_retrieval(query: str) -> list[Document]:
        raise AssertionError(f"Conversation unexpectedly retrieved: {query}")

    monkeypatch.setattr(lgos_rag_module, "_retrieve_documents", unexpected_retrieval)
    stub_decisions(
        monkeypatch,
        [AIMessage(content="respond_direct")],
    )
    direct_model = FakeListChatModel(responses=[HISTORY_ANSWER])
    monkeypatch.setattr(lgos_rag_module, "_chat_model", lambda: direct_model)
    request = make_request(
        "lgos-rag",
        messages=[
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am the LGOS RAG assistant."},
            {"role": "user", "content": "What did I ask you?"},
        ],
    )

    events = [
        event
        async for event in run_langgraph_stream(
            "lgos-rag",
            request.messages,
            lgos_rag_registry(),
            request,
        )
    ]

    assert (
        "".join(event for event in events if isinstance(event, str)) == HISTORY_ANSWER
    )
    assert not [event for event in events if not isinstance(event, str)]


@pytest.mark.anyio
async def test_lgos_rag_stops_after_bounded_irrelevant_retrievals(
    make_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []

    async def retrieve_documents(query: str) -> list[Document]:
        queries.append(query)
        return [source_document("Unrelated content", "Irrelevant")]

    monkeypatch.setattr(lgos_rag_module, "_retrieve_documents", retrieve_documents)
    stub_decisions(
        monkeypatch,
        [
            tool_call("weak query", "retrieve-1"),
            tool_call(REWRITTEN_QUESTION, "retrieve-2"),
        ],
    )
    stub_grades(monkeypatch, ["no", "no"])
    refusal = "I cannot answer that from the available LGOS documentation."
    rewrite_model = FakeListChatModel(responses=[REWRITTEN_QUESTION])
    refusal_model = FakeListChatModel(responses=[refusal])
    monkeypatch.setattr(lgos_rag_module, "_internal_chat_model", lambda: rewrite_model)
    monkeypatch.setattr(lgos_rag_module, "_chat_model", lambda: refusal_model)
    request = make_request("lgos-rag", content="How do I call it?")

    events = [
        event
        async for event in run_langgraph_stream(
            "lgos-rag",
            request.messages,
            lgos_rag_registry(),
            request,
        )
    ]

    assert queries == ["weak query", REWRITTEN_QUESTION]
    assert "".join(event for event in events if isinstance(event, str)) == refusal
    assert not [event for event in events if not isinstance(event, str)]
