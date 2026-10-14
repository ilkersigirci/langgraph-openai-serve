"""End-to-end quality of lgos-rag over the published docs of the installed LGOS."""

import asyncio
import re
from typing import NamedTuple

import httpx2
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve.graph.citations import citations_from_message

from lgos_demo_api.graphs.lgos_rag import (
    LgosRagSettings,
    Scope,
    create_lgos_rag_graph,
    docs_release,
    docs_url,
)

pytestmark = pytest.mark.integration

REFERENCE = "reference/"
COMPATIBILITY = "explanation/openai-compatibility/"
SERVER = "how-to-guides/server/"
BACKGROUND = "how-to-guides/background-responses/"
LOGGING = "how-to-guides/production-logging/"
FILES = "how-to-guides/file-inputs/"
AUTH = "how-to-guides/authentication/"
CLIENTS = "tutorials/openai-clients/"
INTERRUPTS = (COMPATIBILITY, CLIENTS, REFERENCE)
# The demo's LiteLLM page shows registering LGOS models behind that proxy.
PROXIES = ("how-to-guides/openai-proxies/", "demo/litellm-sync/")
# Model sampling flips a case between runs, while a regression fails many;
# three runs of the graph failed one case at most.
MAX_FAILURES = 3


class Case(NamedTuple):
    question: str
    # Pages that answer each part of the question; each part needs one of them
    # in the results. No parts means the graph must not search.
    parts: tuple[set[str], ...] = ()
    # Earlier user and assistant turns, alternating.
    history: tuple[str, ...] = ()
    scope: Scope = "all"


def _one(*pages: str) -> tuple[set[str]]:
    return (set(pages),)


CASES = (
    # Questions in a user's words.
    Case("How do I install langgraph-openai-serve?", _one("getting-started/")),
    Case(
        "How do I expose my LangGraph graph as an OpenAI model?",
        _one("getting-started/", "tutorials/custom-graphs/", REFERENCE),
    ),
    Case(
        "How do I run Responses in the background and poll them?",
        _one(BACKGROUND, COMPATIBILITY),
    ),
    Case("How do I protect the API with bearer tokens?", _one(AUTH)),
    Case(
        "How can a client pass per-request runtime settings to a graph?",
        _one("how-to-guides/langgraph-runtime-settings/", COMPATIBILITY),
    ),
    Case(
        "How does a client answer a graph interrupt and resume it?",
        _one(*INTERRUPTS),
    ),
    Case("How do I enable Langfuse tracing?", _one(REFERENCE, LOGGING, SERVER)),
    Case(
        "How do I hide internal model output from the client stream?",
        _one("explanation/langgraph-integration/", COMPATIBILITY),
    ),
    Case(
        "How do I publish a status update from a graph node?",
        _one(REFERENCE, COMPATIBILITY, "tutorials/custom-graphs/"),
    ),
    Case(
        "How do I add URL citations to the assistant answer?",
        _one(REFERENCE, COMPATIBILITY, "demo/graphs/events-and-citations/"),
    ),
    Case("How do I accept file uploads in my graph?", _one(FILES)),
    Case(
        "How do I put LGOS behind an OpenAI-compatible proxy like LiteLLM?",
        _one(*PROXIES),
    ),
    Case(
        "How do I generate a new project with the starter template?",
        _one("starter-template/"),
    ),
    Case("How do I correlate logs with request IDs?", _one(LOGGING)),
    Case("What does the lgos worker command run?", _one(SERVER, REFERENCE)),
    Case(
        "What happens to a running graph when the client disconnects?",
        _one("explanation/langgraph-integration/"),
    ),
    Case(
        "Does Chat Completions streaming report token usage?",
        _one(COMPATIBILITY, REFERENCE),
    ),
    Case(
        "How do I configure PostgreSQL persistence for the server?",
        _one(SERVER, REFERENCE),
    ),
    Case(
        "How do I start the demo stack with Docker Compose?",
        _one("demo/docker/", "demo/"),
    ),
    Case("How does Open WebUI show interrupt reviews?", _one("demo/open-webui/")),
    Case(
        "How do I call LGOS with the OpenAI Python SDK?",
        _one(CLIENTS, "getting-started/"),
    ),
    Case(
        "How do background create retries avoid duplicate runs?",
        _one(COMPATIBILITY, BACKGROUND, REFERENCE),
    ),
    # Exact names.
    Case("What does LGOS_INTERRUPT_TTL_MINUTES control?", _one(SERVER, REFERENCE)),
    Case(
        "Why would resuming an interrupt return a run_busy 409 error?",
        _one(REFERENCE, COMPATIBILITY),
    ),
    Case("When is RunBusyError raised?", _one(REFERENCE)),
    Case("What does LGOS_BACKGROUND accept?", _one(SERVER, REFERENCE, BACKGROUND)),
    Case("What is GraphFeature.FILE_INPUTS for?", _one(REFERENCE, FILES)),
    Case("What does citation_slice do?", _one(REFERENCE)),
    Case(
        "What does LGOS_INTERRUPT_SWEEP_INTERVAL_MINUTES configure?",
        _one(SERVER, REFERENCE),
    ),
    Case(
        "What is lgos.server_tools in the model list?",
        _one(COMPATIBILITY, REFERENCE),
    ),
    Case("What does LGOS_CORS_ORIGINS set?", _one(SERVER, "demo/reference/")),
    Case(
        "When do I need InMemoryRunCoordinator?",
        _one(REFERENCE, "tutorials/custom-graphs/"),
    ),
    Case(
        "How does the display_file client tool work?",
        _one(FILES, COMPATIBILITY, REFERENCE),
    ),
    Case(
        "What is the Idempotency-Key header for?",
        _one(COMPATIBILITY, BACKGROUND, REFERENCE),
    ),
    # Follow-ups that only make sense with the conversation.
    Case(
        "How do I point it at a self-hosted instance?",
        _one(REFERENCE, "demo/self-hosted/"),
        history=(
            "How do I enable Langfuse tracing?",
            (
                "Install the tracing extra and set LGOS_ENABLE_LANGFUSE=True with "
                "your Langfuse keys."
            ),
        ),
    ),
    Case(
        "How do clients poll it?",
        _one(BACKGROUND, COMPATIBILITY, REFERENCE),
        history=(
            "How do I run responses in the background?",
            "Set LGOS_BACKGROUND=hatchet and run the lgos worker process.",
        ),
    ),
    Case(
        "How long are paused runs kept?",
        _one(SERVER, REFERENCE, "demo/graphs/interruptible-approval/"),
        history=(
            "How do interrupts work in LGOS?",
            (
                "A paused graph returns lgos_interrupt function calls that the "
                "client answers."
            ),
        ),
    ),
    Case(
        "Does that work with the OpenAI SDK's api_key?",
        _one(AUTH, CLIENTS),
        history=(
            "Can I add authentication?",
            "Yes. LGOS can require bearer tokens on its routes.",
        ),
    ),
    Case(
        "And how do I show a generated file back to the user?",
        _one(FILES, COMPATIBILITY),
        history=(
            "How do I accept file uploads?",
            "Declare GraphFeature.FILE_INPUTS and read the file blocks.",
        ),
    ),
    Case(
        "How do I generate it?",
        _one("starter-template/"),
        history=(
            "What is the starter template?",
            "A Cookiecutter project that lgos serve runs, with tests and Docker.",
        ),
    ),
    # Questions with several parts.
    Case(
        "How do I enable Langfuse tracing, and how do I correlate logs with "
        "request IDs?",
        ({REFERENCE, LOGGING, SERVER}, {LOGGING}),
    ),
    Case(
        "How do I protect the API with bearer tokens and accept file uploads?",
        ({AUTH}, {FILES}),
    ),
    Case(
        "What does LGOS_INTERRUPT_TTL_MINUTES do, and how do I run responses in "
        "the background?",
        ({SERVER, REFERENCE}, {BACKGROUND, COMPATIBILITY}),
    ),
    Case(
        "How do I generate a starter project and start the demo with Docker Compose?",
        ({"starter-template/"}, {"demo/docker/", "demo/"}),
    ),
    Case(
        "How do I put LGOS behind LiteLLM, and how does a client resume an interrupt?",
        (set(PROXIES), set(INTERRUPTS)),
    ),
    # Scoped search: the library scope skips the demo's LiteLLM page.
    Case(
        "How do I put LGOS behind LiteLLM?",
        _one("how-to-guides/openai-proxies/"),
        scope="library",
    ),
    Case(
        "How do I start the stack with Docker Compose?",
        _one("demo/docker/", "demo/"),
        scope="demo",
    ),
    # No search: greetings, the assistant, the conversation, unrelated topics.
    Case("Hi!"),
    Case("Who are you?"),
    Case(
        "Thanks, that helped.",
        history=("How do I install LGOS?", "Run uv add langgraph-openai-serve."),
    ),
    Case(
        "What did I ask you before?",
        history=("How do I install LGOS?", "Run uv add langgraph-openai-serve."),
    ),
    Case("What is the capital of France?"),
    Case("Write a haiku about autumn."),
    Case("How do I center a div in CSS?"),
    Case("Who won the 2018 FIFA World Cup?"),
)

_LINK = re.compile(r"(?<!!)\[[^]\r\n]+\]\((?P<url>[^)\r\n]+)\)")


def _messages(case: Case) -> list[BaseMessage]:
    turns = [*case.history, case.question]
    return [
        HumanMessage(content=text) if position % 2 == 0 else AIMessage(content=text)
        for position, text in enumerate(turns)
    ]


def _page(url: str, base_url: str) -> str:
    return url.removeprefix(base_url).partition("#")[0]


async def _failures(graph: CompiledStateGraph, case: Case, base_url: str) -> list[str]:
    output = await graph.ainvoke(
        {"messages": _messages(case)}, context=LgosRagSettings(scope=case.scope)
    )
    # LangGraph omits fields that kept their defaults, such as an empty plan.
    queries = output.get("queries", [])
    urls = {str(document.metadata["url"]) for document in output.get("documents", [])}
    answer = output["messages"][-1]
    links = {match["url"] for match in _LINK.finditer(answer.text)}
    if not case.parts:
        return [f"searched {queries}"] if queries else []
    if not queries:
        return ["did not search"]
    found = {_page(url, base_url) for url in urls}
    if case.scope != "all" and any(
        page.startswith("demo/") != (case.scope == "demo") for page in found
    ):
        return [f"searched outside the {case.scope} docs: {sorted(found)}"]
    failures = [
        f"no answering page for {sorted(pages)} in {sorted(found)}"
        for pages in case.parts
        if not pages & found
    ]
    # Links copied from the retrieved text are real; any other link is invented.
    copied = {
        match["url"]
        for document in output.get("documents", [])
        for match in _LINK.finditer(document.page_content)
    }
    if invented := links - urls - copied:
        failures.append(f"invented links: {sorted(invented)}")
    if not citations_from_message(answer):
        failures.append("cites no retrieved section")
    return failures


async def test_cases_search_cite_and_scope_as_expected() -> None:
    graph = create_lgos_rag_graph()
    async with httpx2.AsyncClient(follow_redirects=True, timeout=30) as client:
        base_url = docs_url(await docs_release(client))
    # The first case builds the index; the rest run concurrently.
    await _failures(graph, CASES[0], base_url)
    limit = asyncio.Semaphore(8)

    async def run(case: Case) -> list[str]:
        async with limit:
            return await _failures(graph, case, base_url)

    results = await asyncio.gather(*(run(case) for case in CASES))

    failed = {
        case.question: failures
        for case, failures in zip(CASES, results, strict=True)
        if failures
    }
    assert len(failed) <= MAX_FAILURES, failed
