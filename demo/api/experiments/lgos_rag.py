"""
lgos-rag's evaluation as a Langfuse experiment over the published docs.

Each case is a conversation with the pages that answer it, or the reply shape
expected instead. `CaseInput` and `CaseExpectation` define the case format for
the Git-tracked cases below and for cases curated in Langfuse. The checks score
routing, reply shape, retrieval, scope, links, and citations, and `_shared` runs
them in Langfuse or, without it, locally.
"""

import re
from importlib.metadata import version
from typing import Any, Literal, NamedTuple

from _shared import main, run, score
from langchain_core.messages import convert_to_messages
from langfuse import Evaluation, RunnerContext
from langfuse.experiment import ExperimentResult, LocalExperimentItem
from langfuse.langchain import CallbackHandler
from langgraph_openai_serve.graph.citations import citations_from_message
from pydantic import BaseModel, ConfigDict, Field

from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs.lgos_rag import (
    DOCS_SITE_URL,
    LgosRagSettings,
    Scope,
    create_lgos_rag_graph,
)

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


# The reply's shape, checked by REPLY_CHECKS instead of the results.
Reply = Literal["scope_hint", "clarify", "small_talk"]


class Case(NamedTuple):
    """A Git-tracked case in brief; `_item` writes it in the case format."""

    question: str
    # CaseExpectation.parts, with each part's pages as a set.
    parts: tuple[set[str], ...] = ()
    # Earlier user and assistant turns, alternating.
    history: tuple[str, ...] = ()
    scope: Scope = "all"
    reply: Reply | None = None


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
    Case(
        "What does a client receive when a graph raises GraphError?",
        _one(COMPATIBILITY),
    ),
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
    # The reply's shape: a scope hint after a restricted miss, and a clarifying
    # question for a message too vague to answer.
    # Only the demo pages document this variable.
    Case(
        "What does DEMO_GRAPH_GATEWAY_API_KEY set?",
        scope="library",
        reply="scope_hint",
    ),
    Case(
        "Why is it failing?",
        reply="clarify",
    ),
    # No search, but no refusal either: greetings, the assistant, the
    # conversation.
    Case("Hi!", reply="small_talk"),
    Case("Who are you?", reply="small_talk"),
    Case(
        "Thanks, that helped.",
        history=("How do I install LGOS?", "Run uv add langgraph-openai-serve."),
        reply="small_talk",
    ),
    Case(
        "What did I ask you before?",
        history=("How do I install LGOS?", "Run uv add langgraph-openai-serve."),
        reply="small_talk",
    ),
    # No search: unrelated topics.
    Case("What is the capital of France?"),
    Case("Write a haiku about autumn."),
    Case("How do I center a div in CSS?"),
    Case("Who won the 2018 FIFA World Cup?"),
)

_LINK = re.compile(r"(?<!!)\[[^]\r\n]+\]\((?P<url>[^)\r\n]+)\)")
REPLY_CHECKS = {
    "scope_hint": lambda text: "docs scope" in text.lower(),
    "clarify": lambda text: "?" in text and "only answer" not in text.lower(),
    # The graph refuses other requests by saying it only answers LGOS questions.
    "small_talk": lambda text: "only answer" not in text.lower(),
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class CaseInput(BaseModel):
    """A conversation that ends with the user's message, and the graph settings."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)
    settings: LgosRagSettings = LgosRagSettings()


class CaseExpectation(BaseModel):
    """What the graph's reply to the conversation must do."""

    model_config = ConfigDict(extra="forbid")

    parts: list[list[str]] = Field(
        description=(
            "For each part of the question, the docs pages that answer it, as "
            "paths such as how-to-guides/authentication/; the search must find "
            "one page of each part. Empty when no page in the scope answers it."
        )
    )
    reply: Reply | None = Field(
        default=None,
        description=(
            "scope_hint: the reply suggests the Docs scope All after a search of "
            "a restricted scope. clarify: the reply asks a clarifying question. "
            "small_talk: the reply answers a greeting, thanks, or a question "
            "about the assistant or the conversation instead of refusing."
        ),
    )


def _item(case: Case) -> LocalExperimentItem:
    turns = (*case.history, case.question)
    messages = [
        Message(role="user" if position % 2 == 0 else "assistant", content=text)
        for position, text in enumerate(turns)
    ]
    expectation = CaseExpectation(
        parts=[sorted(pages) for pages in case.parts], reply=case.reply
    )
    return {
        "input": CaseInput(
            messages=messages, settings=LgosRagSettings(scope=case.scope)
        ).model_dump(mode="json"),
        "expected_output": expectation.model_dump(mode="json"),
    }


def _question(case_input: dict[str, Any]) -> str:
    """Name a case by its last message; the Git-tracked cases keep them unique."""
    return case_input["messages"][-1]["content"]


ITEMS = [_item(case) for case in CASES]
_GRAPH = create_lgos_rag_graph()


async def _answer(*, item: LocalExperimentItem, **_kwargs: Any) -> dict[str, Any]:
    case = CaseInput.model_validate(item["input"])
    state = await _GRAPH.ainvoke(
        {
            "messages": convert_to_messages(
                [(turn.role, turn.content) for turn in case.messages]
            )
        },
        # The handler records the graph's calls in the case's trace.
        config={"callbacks": [CallbackHandler()]},
        context=case.settings,
    )
    # LangGraph omits fields no node wrote, such as documents when the plan has
    # no queries.
    documents = state.get("documents", [])
    message = state["messages"][-1]
    return {
        "answer": message.text,
        "queries": state.get("queries", []),
        "sources": [str(document.metadata["url"]) for document in documents],
        "source_links": sorted(
            {
                match["url"]
                for document in documents
                for match in _LINK.finditer(document.page_content)
            }
        ),
        "citations": [citation["url"] for citation in citations_from_message(message)],
    }


def _page(url: str) -> str:
    """Return a docs URL's page path without its release, such as reference/."""
    return url.removeprefix(f"{DOCS_SITE_URL}/").partition("/")[2].partition("#")[0]


def checks(
    *,
    input: dict[str, Any],  # ruff: ignore[builtin-argument-shadowing] - Langfuse passes the item input by this name.
    output: dict[str, Any],
    expected_output: dict[str, Any],
    **_kwargs: Any,
) -> list[Evaluation]:
    """Score each check that applies to the case as passed or failed."""
    case = CaseInput.model_validate(input)
    expected = CaseExpectation.model_validate(expected_output)
    answer = output["answer"]
    parts = [set(pages) for pages in expected.parts]
    reply = expected.reply
    queries = output["queries"]
    # A scope hint follows a search that found nothing in the selected scope.
    searches = bool(parts) or reply == "scope_hint"
    scores = [
        score(
            "route",
            passed=bool(queries) == searches,
            comment=f"queries: {queries}",
        )
    ]
    if reply:
        scores.append(
            score("reply", passed=REPLY_CHECKS[reply](answer), comment=answer)
        )
    if not parts or not queries:
        return scores
    found = {_page(url) for url in output["sources"]}
    missing = [sorted(pages) for pages in parts if not pages & found]
    scores.append(
        score(
            "retrieval",
            passed=not missing,
            comment=f"missing {missing} in {sorted(found)}" if missing else None,
        )
    )
    if (scope := case.settings.scope) != "all":
        outside = sorted(
            page for page in found if page.startswith("demo/") != (scope == "demo")
        )
        scores.append(
            score(
                "scope",
                passed=not outside,
                comment=f"outside: {outside}" if outside else None,
            )
        )
    # Links copied from the retrieved text are real; any other link is invented.
    links = {match["url"] for match in _LINK.finditer(answer)}
    invented = sorted(links - set(output["sources"]) - set(output["source_links"]))
    scores.extend(
        [
            score(
                "links",
                passed=not invented,
                comment=f"invented: {invented}" if invented else None,
            ),
            score("citation", passed=bool(output["citations"])),
        ]
    )
    return scores


def experiment(context: RunnerContext) -> ExperimentResult:
    """Run lgos-rag's cases; the entry point of the Langfuse experiment runner."""
    return run(
        context,
        name="lgos-rag",
        description="Routing, retrieval, scope, links, and citations per case.",
        items=ITEMS,
        case_id=_question,
        input_schema=CaseInput,
        expected_output_schema=CaseExpectation,
        task=_answer,
        evaluators=[checks],
        # Three runs of the graph failed one case at most.
        max_failures=3,
        metadata={
            # Change it with a check, so runs graded differently are not compared.
            "evaluator_version": "2",
            # lgos-rag answers from this release's docs.
            "lgos": version("langgraph-openai-serve"),
            "chat_model": settings.OPENAI_CHAT_COMPLETIONS_MODEL,
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        },
    )


if __name__ == "__main__":
    main(experiment, ITEMS)
