"""
lgos-rag's evaluation as a Langfuse experiment over the published docs.

The cases below are the source of truth. With Langfuse configured, each run
syncs them into the hosted Langfuse dataset, runs the graph on every item, and
records one pass-or-fail score per check, so Langfuse compares runs case by
case. Without Langfuse, the same cases and checks run on local data and record
nothing, which suits CI. Either way, too many failing cases fail the run.

The file follows the Langfuse experiment runner convention, so the
langfuse/experiment-action GitHub Action can run it unchanged;
`just demo/experiment-rag` runs it locally.
"""

import hashlib
import json
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] - reads the checkout's commit for run metadata.
from importlib.metadata import version
from typing import Any, Literal, NamedTuple

from langchain_core.messages import convert_to_messages
from langfuse import (
    Evaluation,
    Langfuse,
    RegressionError,
    RunnerContext,
    get_client,
)
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import (
    ExperimentData,
    ExperimentItem,
    ExperimentItemResult,
    ExperimentResult,
)
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

DATASET = "lgos-rag"
# Bump when a check changes, so runs graded differently are not compared.
EVALUATOR_VERSION = "2"

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
    # A check on the reply instead of the search, named in REPLY_CHECKS.
    reply: Literal["scope_hint", "clarify"] | None = None


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
REPLY_CHECKS = {
    "scope_hint": lambda text: "docs scope" in text.lower(),
    "clarify": lambda text: "?" in text and "only answer" not in text.lower(),
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class CaseInput(BaseModel):
    """A dataset item's input: the conversation and the search scope."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)
    scope: Scope


class CaseExpectation(BaseModel):
    """A dataset item's expected output: answering pages or a reply shape."""

    model_config = ConfigDict(extra="forbid")

    parts: list[list[str]]
    reply: Literal["scope_hint", "clarify"] | None


def _item(case: Case) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return a case's dataset item ID, input, and expected output."""
    turns = [*case.history, case.question]
    item_input = CaseInput(
        messages=[
            Message(role="user" if position % 2 == 0 else "assistant", content=text)
            for position, text in enumerate(turns)
        ],
        scope=case.scope,
    ).model_dump()
    expected = CaseExpectation(
        parts=[sorted(pages) for pages in case.parts], reply=case.reply
    ).model_dump()
    digest = hashlib.sha256(json.dumps(item_input, sort_keys=True).encode())
    # Langfuse item IDs are unique across a project's datasets.
    return f"{DATASET}-{digest.hexdigest()[:16]}", item_input, expected


def _sync_dataset(langfuse: Langfuse) -> None:
    """Make the hosted dataset's active items exactly this file's cases."""
    # Creating an existing dataset updates it, so its schemas follow the models.
    langfuse.create_dataset(
        name=DATASET,
        description="Managed by demo/api/experiments/lgos_rag.py.",
        input_schema=CaseInput.model_json_schema(),
        expected_output_schema=CaseExpectation.model_json_schema(),
    )
    current = {item.id: item for item in langfuse.get_dataset(DATASET).items}
    cases = {
        item_id: (item_input, expected)
        for item_id, item_input, expected in map(_item, CASES)
    }
    # Rewriting an unchanged item would add a dataset version for nothing.
    for item_id, (item_input, expected) in cases.items():
        item = current.get(item_id)
        if item is None or (item.input, item.expected_output) != (item_input, expected):
            langfuse.create_dataset_item(
                dataset_name=DATASET,
                id=item_id,
                input=item_input,
                expected_output=expected,
            )
    for item_id, item in current.items():
        if item_id not in cases and item.status == DatasetStatus.ACTIVE:
            langfuse.create_dataset_item(
                dataset_name=DATASET,
                id=item_id,
                input=item.input,
                expected_output=item.expected_output,
                status=DatasetStatus.ARCHIVED,
            )


def _page(url: str) -> str:
    """Return a docs URL's page path without its release, such as reference/."""
    return url.removeprefix(f"{DOCS_SITE_URL}/").partition("/")[2].partition("#")[0]


def _score(name: str, *, passed: bool, comment: str | None = None) -> Evaluation:
    return Evaluation(
        name=name, value=float(passed), data_type="BOOLEAN", comment=comment
    )


def _checks(
    *,
    input: dict[str, Any],  # ruff: ignore[builtin-argument-shadowing] - Langfuse passes the item input by this name.
    output: dict[str, Any],
    expected_output: dict[str, Any],
    **_kwargs: Any,
) -> list[Evaluation]:
    """Score each check that applies to the case as passed or failed."""
    expected = CaseExpectation.model_validate(expected_output)
    answer = output["answer"]
    if expected.reply:
        return [
            _score("reply", passed=REPLY_CHECKS[expected.reply](answer), comment=answer)
        ]
    parts = [set(pages) for pages in expected.parts]
    queries = output["queries"]
    scores = [
        _score(
            "route", passed=bool(queries) == bool(parts), comment=f"queries: {queries}"
        )
    ]
    if not parts or not queries:
        return scores
    found = {_page(url) for url in output["sources"]}
    missing = [sorted(pages) for pages in parts if not pages & found]
    scores.append(
        _score(
            "retrieval",
            passed=not missing,
            comment=f"missing {missing} in {sorted(found)}" if missing else None,
        )
    )
    if (scope := CaseInput.model_validate(input).scope) != "all":
        outside = sorted(
            page for page in found if page.startswith("demo/") != (scope == "demo")
        )
        scores.append(
            _score(
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
            _score(
                "links",
                passed=not invented,
                comment=f"invented: {invented}" if invented else None,
            ),
            _score("citation", passed=bool(output["citations"])),
        ]
    )
    return scores


def _all_checks_passed(*, evaluations: list[Evaluation], **_kwargs: Any) -> Evaluation:
    """Score a case as passed when it has checks and passed every one."""
    failed = [score.name for score in evaluations if score.value != 1]
    return _score(
        "passed",
        passed=bool(evaluations) and not failed,
        comment=f"failed: {failed}" if failed else None,
    )


def _input(item: ExperimentItem) -> CaseInput:
    # Hosted dataset items are objects; local data items are dictionaries.
    raw = item.input if isinstance(item, DatasetItem) else item["input"]
    return CaseInput.model_validate(raw)


def _passed(result: ExperimentItemResult) -> bool:
    # A failed task or a crashed evaluator leaves no passed score.
    return any(
        score.name == "passed" and score.value == 1 for score in result.evaluations
    )


def _pass_rate(
    *, item_results: list[ExperimentItemResult], **_kwargs: Any
) -> Evaluation:
    passed = sum(map(_passed, item_results))
    return Evaluation(
        name="pass_rate",
        value=passed / len(item_results),
        comment=f"{passed} of {len(item_results)} cases passed",
    )


def _metadata() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # ruff: ignore[start-process-with-partial-path] - git from PATH, fixed arguments.
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "commit": commit or "unknown",
        "lgos": version("langgraph-openai-serve"),
        "chat_model": settings.OPENAI_CHAT_COMPLETIONS_MODEL,
        "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "evaluators": EVALUATOR_VERSION,
    }


def _run(context: RunnerContext, data: ExperimentData) -> ExperimentResult:
    """Run the cases, score them, and fail when too many cases fail."""
    graph = create_lgos_rag_graph()

    async def answer(*, item: ExperimentItem, **_kwargs: Any) -> dict[str, Any]:
        case_input = _input(item)
        state = await graph.ainvoke(
            {
                "messages": convert_to_messages(
                    [message.model_dump() for message in case_input.messages]
                )
            },
            config={"callbacks": [CallbackHandler()]},
            context=LgosRagSettings(scope=case_input.scope),
        )
        # LangGraph omits fields that kept their defaults, such as an empty plan.
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
            "citations": [
                citation["url"] for citation in citations_from_message(message)
            ],
        }

    result = context.run_experiment(
        name="lgos-rag",
        description="Routing, retrieval, scope, links, and citations per case.",
        data=data,
        task=answer,
        evaluators=[_checks],
        composite_evaluator=_all_checks_passed,
        run_evaluators=[_pass_rate],
        max_concurrency=8,
        metadata=_metadata(),
    )
    failed = {
        _input(item_result.item).messages[-1].content: [
            f"{score.name}: {score.comment}"
            for score in item_result.evaluations
            if score.name != "passed" and score.value != 1
        ]
        or ["no checks scored"]
        for item_result in result.item_results
        if not _passed(item_result)
    }
    if len(failed) > MAX_FAILURES:
        raise RegressionError(
            result=result,
            metric="failed_cases",
            value=len(failed),
            threshold=MAX_FAILURES,
            message=f"{len(failed)} cases failed: {failed}",
        )
    return result


def experiment(context: RunnerContext) -> ExperimentResult:
    """Sync the hosted dataset, then run, trace, and score its items in Langfuse."""
    _sync_dataset(context.client)
    return _run(context, context.client.get_dataset(DATASET).items)


if __name__ == "__main__":
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        langfuse = get_client()
        try:
            outcome = experiment(RunnerContext(client=langfuse))
        finally:
            langfuse.flush()
    else:
        # Without Langfuse, the same cases and checks run on local data, and
        # nothing is recorded; the gate still fails the run.
        print("Langfuse is not configured; running the cases without recording.")
        offline = RunnerContext(client=Langfuse(tracing_enabled=False))
        outcome = _run(
            offline,
            [
                {"input": item_input, "expected_output": expected}
                for _, item_input, expected in map(_item, CASES)
            ],
        )
    print(outcome.format())
