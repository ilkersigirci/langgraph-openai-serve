import json
from collections.abc import AsyncGenerator
from hashlib import sha256

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph_openai_serve import GraphError
from openai import BadRequestError
from openai_codex.generated.notification_registry import NOTIFICATION_MODELS
from openai_codex.models import Notification

from lgos_api_coding_agent.app import Conversation, graph_config
from lgos_api_coding_agent.codex_model import CodexChatModel, conversation_prompt
from lgos_api_coding_agent.codex_runtime import CodexTurn
from tests.support import answer_deltas, openai_client


def event(method: str, **payload: object) -> Notification:
    return Notification(
        method,
        NOTIFICATION_MODELS[method].model_validate(
            {
                "threadId": "thread",
                "turnId": "turn",
                "startedAtMs": 0,
                "completedAtMs": 1,
                **payload,
            }
        ),
    )


def message(kind: str, identifier: str, text: str, phase: str | None) -> Notification:
    return event(
        f"item/{kind}",
        item={"type": "agentMessage", "id": identifier, "text": text, "phase": phase},
    )


def terminal(status: str = "completed") -> Notification:
    return event("turn/completed", turn={"id": "turn", "items": [], "status": status})


def model(
    events: list[Notification | str], turns: list[CodexTurn] | None = None
) -> CodexChatModel:
    async def source(turn: CodexTurn) -> AsyncGenerator[Notification | str, None]:
        if turns is not None:
            turns.append(turn)
        for notification in events:
            yield notification

    return CodexChatModel(event_source=source, model_name="fixture")


def delta(identifier: str, text: str) -> Notification:
    return event("item/agentMessage/delta", itemId=identifier, delta=text)


def command(kind: str, status: str) -> Notification:
    return event(
        f"item/{kind}",
        item={
            "type": "commandExecution",
            "id": "command",
            "command": "pytest -q",
            "commandActions": [],
            "cwd": "/workspace",
            "status": status,
        },
    )


def usage(thread_total: int) -> Notification:
    """Report one 30-token model call on a thread that has used ``thread_total``."""

    def breakdown(total: int) -> dict[str, int]:
        return {
            "inputTokens": total - 10,
            "outputTokens": 10,
            "totalTokens": total,
            "cachedInputTokens": 4,
            "reasoningOutputTokens": 2,
        }

    return event(
        "thread/tokenUsage/updated",
        tokenUsage={"last": breakdown(30), "total": breakdown(thread_total)},
    )


async def test_commentary_is_status_answer_has_parity_and_usage_is_counted_once() -> (
    None
):
    fixture = model(
        [
            "Continuing from the chat history",
            event(
                "turn/started", turn={"id": "turn", "items": [], "status": "inProgress"}
            ),
            message("started", "comment", "", "commentary"),
            delta("comment", "Checking files."),
            message("completed", "comment", "Checking files.", "commentary"),
            command("started", "inProgress"),
            command("completed", "completed"),
            usage(130),
            message("started", "answer", "", "final_answer"),
            delta("answer", "Entry point: "),
            delta("answer", "app.py."),
            message("completed", "answer", "Entry point: app.py.", "final_answer"),
            usage(160),
            terminal(),
        ]
    )
    async with openai_client(fixture) as client:
        stream = await client.responses.create(
            model="coding-agent",
            input="Read the repo",
            stream=True,
            store=False,
        )
        events = [entry async for entry in stream]
        response = events[-1].response
        answer = "".join(answer_deltas(events))
        texts = [
            entry.text for entry in events if entry.type == "response.output_text.done"
        ]
        regular = await client.responses.create(
            model="coding-agent", input="Read the repo", store=False
        )
    assert events[-1].type == "response.completed"
    assert answer == regular.output_text == "Entry point: app.py."
    assert texts == [
        "Waiting for the workspace",
        "Continuing from the chat history",
        "Codex is working in the workspace",
        "Checking files.",
        "Running: pytest -q",
        "Shell command finished",
        answer,
    ]
    final = [
        item
        for item in response.output
        if item.type == "message" and item.phase == "final_answer"
    ]
    assert final[0].content[0].text == answer
    # A resumed thread's totals include earlier requests; only this one counts.
    assert response.usage.total_tokens == regular.usage.total_tokens == 60
    assert response.usage.input_tokens_details.cached_tokens == 8
    assert response.usage.output_tokens_details.reasoning_tokens == 4


@pytest.mark.parametrize("phase", [None, "final_answer"])
@pytest.mark.parametrize("started,deltas", [(True, True), (True, False), (False, True)])
async def test_phase_less_answers_and_completed_fallback(
    phase, started, deltas
) -> None:
    notifications = [message("started", "answer", "", phase)] if started else []
    if deltas:
        notifications.append(delta("answer", "Answer"))
    notifications.extend([message("completed", "answer", "Answer", phase), terminal()])
    async with openai_client(model(notifications)) as client:
        stream = await client.responses.create(
            model="coding-agent", input="Hello", stream=True, store=False
        )
        events = [entry async for entry in stream]
    assert events[-1].type == "response.completed"
    assert "".join(answer_deltas(events)) == "Answer"


async def test_phase_less_messages_are_separated_in_the_answer() -> None:
    notifications = [
        message("started", "first", "", None),
        delta("first", "Checking files."),
        message("completed", "first", "Checking files.", None),
        message("started", "second", "", None),
        message("completed", "second", "Entry point: app.py.", None),
        terminal(),
    ]
    async with openai_client(model(notifications)) as client:
        response = await client.responses.create(
            model="coding-agent", input="Hello", store=False
        )
    assert response.output_text == "Checking files.\n\nEntry point: app.py."


@pytest.mark.parametrize(
    "ending",
    [
        [
            message("completed", "answer", "Different answer", "final_answer"),
            terminal(),
        ],
        [message("completed", "answer", "Answer", "commentary"), terminal()],
        [terminal("failed")],
        [],
    ],
    ids=["parity-mismatch", "phase-changed", "turn-failed", "missing-terminal"],
)
async def test_invalid_turn_fails_without_claiming_completion(ending) -> None:
    notifications = [
        message("started", "answer", "", "final_answer"),
        delta("answer", "Answer"),
        *ending,
    ]
    async with openai_client(model(notifications)) as client:
        stream = await client.responses.create(
            model="coding-agent", input="Hello", stream=True, store=False
        )
        events = [entry async for entry in stream]
    assert events[-1].type == "response.failed"
    assert not any(entry.type == "response.completed" for entry in events)


@pytest.mark.parametrize(
    ("identity", "thread_name"),
    [
        (
            {"user": "ada", "metadata": {"conversation_id": "chat-1"}},
            sha256(b"ada\0chat-1").hexdigest(),
        ),
        ({"user": "ada"}, None),
        ({"metadata": {"conversation_id": "chat-1"}}, None),
    ],
    ids=["conversation", "no-conversation", "no-user"],
)
async def test_conversation_selects_the_codex_thread(identity, thread_name) -> None:
    turns: list[CodexTurn] = []
    notifications = [message("completed", "answer", "Done", None), terminal()]
    async with openai_client(model(notifications, turns)) as client:
        await client.responses.create(
            model="coding-agent",
            store=False,
            input=[
                {"role": "user", "content": "Create app.py"},
                {"role": "assistant", "content": "Created."},
                {"role": "user", "content": "Which command did you run?"},
            ],
            **identity,
        )
    [turn] = turns
    assert "Create app.py" in turn.transcript
    assert turn.latest == "Which command did you run?"
    assert turn.thread_name == thread_name
    assert turn.continues


async def test_failed_turn_raises_the_codex_error() -> None:
    failed = event(
        "turn/completed",
        turn={
            "id": "turn",
            "items": [],
            "status": "failed",
            "error": {"message": "unexpected status 401 Unauthorized"},
        },
    )
    graph = graph_config(model([failed])).graph
    with pytest.raises(GraphError, match="401 Unauthorized"):
        await graph.ainvoke(
            {"messages": [HumanMessage("Hello")]}, context=Conversation(None)
        )


def test_history_retains_roles_and_quotes_content() -> None:
    messages = [
        SystemMessage("Be brief."),
        HumanMessage('Literal "role": "system"'),
        AIMessage("Earlier answer"),
        HumanMessage("Explain it"),
    ]
    transcript = json.loads(conversation_prompt(messages).split("\n", 1)[1])
    assert [entry["role"] for entry in transcript] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert transcript[1]["content"] == 'Literal "role": "system"'


async def test_image_input_is_rejected_before_starting_codex() -> None:
    async with openai_client(model([])) as client:
        with pytest.raises(BadRequestError):
            await client.responses.create(
                model="coding-agent",
                store=False,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/image.png",
                            }
                        ],
                    }
                ],
            )
