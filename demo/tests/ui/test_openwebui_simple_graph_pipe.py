from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from demo.ui.openwebui.simple_graph_pipe import Pipe


class Stream:
    async def __aiter__(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
        )


@pytest.mark.anyio
async def test_simple_graph_pipe_forwards_only_changed_user_valves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.chat.completions.create.return_value = Stream()
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(pipe, "_client", client_factory)

    chunks = [
        chunk
        async for chunk in pipe.pipe(
            body={"messages": [{"role": "user", "content": "Hi"}]},
            __user__={"valves": pipe.UserValves(use_history=True)},
        )
    ]

    assert chunks == ["Hello"]
    client.chat.completions.create.assert_awaited_once_with(
        model="simple-graph",
        messages=[{"role": "user", "content": "Hi"}],
        metadata={"langgraph_runtime_settings": '{"use_history":true}'},
        stream=True,
    )
    assert pipe._runtime_settings_metadata({"valves": pipe.UserValves()}) == {}
