import json
from collections.abc import AsyncIterator
from pathlib import Path

import chainlit as cl
import chainlit.config
import httpx2
import pytest
from chainlit.chat_context import chat_contexts
from chainlit.context import ChainlitContext, init_http_context
from chainlit.step import StepDict
from chainlit.user_session import user_sessions

from lgos_chainlit import audio, chat, clients, display_files
from tests.support import FakeGateway


@pytest.fixture(autouse=True)
def chainlit_app_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    # Chainlit resolves this directory on import; sent elements are written there.
    monkeypatch.setattr(chainlit.config, "FILES_DIRECTORY", tmp_path)


@pytest.fixture
def anyio_backend() -> str:
    """Run the Chainlit test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
async def chainlit_context(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ChainlitContext]:
    """Own a fresh native session with an authenticated demo user."""
    context = init_http_context(user=cl.User(identifier="demo-user"))
    # The HTTP emitter stubs this as a coroutine that is never awaited. Commit
    # ChatSettings.send() values to the session as the WebSocket emitter does.
    monkeypatch.setattr(
        context.emitter,
        "set_chat_settings",
        lambda settings: setattr(context.session, "chat_settings", settings),
    )
    try:
        yield context
    finally:
        user_sessions.pop(context.session.id, None)
        chat_contexts.pop(context.session.id, None)


@pytest.fixture
async def fake_gateway(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeGateway]:
    """Send every demo OpenAI client to one recorded fake gateway."""
    gateway = FakeGateway()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(gateway.handle)
    ) as http:
        for module, name in (
            (clients, "v1_client"),
            (chat, "v1_client"),
            (chat, "responses_client"),
            (display_files, "v1_client"),
            (audio, "v1_client"),
        ):
            client = getattr(module, name)
            monkeypatch.setattr(module, name, client.with_options(http_client=http))
        yield gateway


@pytest.fixture
def task_lists(
    chainlit_context: ChainlitContext,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Select task lists and record each state loaded from Chainlit's file route."""
    monkeypatch.setattr(chat.settings, "STATUS_DISPLAY", "tasklist")
    states: list[dict[str, object]] = []
    send_element = chainlit_context.emitter.send_element

    async def record(element) -> None:
        if element["type"] == "tasklist":
            file = chainlit_context.session.files[element["chainlitKey"]]
            states.append(json.loads(Path(file["path"]).read_text()))
        await send_element(element)

    monkeypatch.setattr(chainlit_context.emitter, "send_element", record)
    return states


@pytest.fixture
def status_steps(
    chainlit_context: ChainlitContext,
    monkeypatch: pytest.MonkeyPatch,
) -> list[StepDict]:
    """Record native tool steps sent and updated in the browser."""
    states: list[StepDict] = []
    monkeypatch.setattr(chainlit.config.config.ui, "cot", "tool_call")

    async def record(step: StepDict) -> None:
        if step["type"] == "tool":
            states.append(step.copy())

    monkeypatch.setattr(chainlit_context.emitter, "send_step", record)
    monkeypatch.setattr(chainlit_context.emitter, "update_step", record)
    return states
