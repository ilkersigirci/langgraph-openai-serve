from collections.abc import AsyncIterator, Sequence
from importlib.metadata import version
from typing import Any, Self

import httpx2
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient, MockTransport
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph_openai_serve import GraphConfig, GraphRequest
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.server import ServerResources, ServerSettings, create_app
from openai import AsyncOpenAI, BadRequestError
from tests.support import ModelProvider, model_response, text_chunks

from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs import server_tool
from lgos_demo_api.graphs.simple import SimpleContext
from lgos_demo_api.registry import open_registry
from lgos_demo_api.utils.web_search import WebSearchResult

DOCUMENTED_MODEL_IDS = {
    "background-interrupt",
    "background-mock",
    "mcp-postgres",
    "citation-events",
    "custom-input-output-context",
    "deep-research",
    "file-input",
    "interruptible-approval",
    "lgos-rag",
    "mcp-mock",
    "persistent-plot-agent",
    "multi-node-streaming",
    "nested-subgraphs",
    "response-outcomes",
    "simple-graph",
    "server-tool",
    "simple-graph-external-tools",
    "status-events",
    "streaming-long-mock",
}


class SearchModel(FakeMessagesListChatModel):
    """Check that a required search binds exactly the expected tools."""

    expected_tools: list[Any]

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Self:
        assert list(tools) == self.expected_tools
        assert kwargs["tool_choice"] == "required"
        return self


def _rebuild_server_tool_graph(demo_app: FastAPI) -> None:
    current = demo_app.state.graph_registry.get_graph("server-tool")
    config = GraphConfig.model_validate(
        {name: getattr(current, name) for name in GraphConfig.model_fields}
        | {"graph": server_tool.create_server_tool_graph()}
    )
    demo_app.state.graph_registry.graphs["server-tool"] = config


@pytest.fixture
async def demo_app() -> AsyncIterator[FastAPI]:
    app = create_app(
        open_registry,
        # Explicit values win over LGOS_* variables that just loads from demo/.env.
        settings=ServerSettings(POSTGRES_URI=None, BACKGROUND="none"),
    )
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def openai_client(demo_app: FastAPI) -> AsyncIterator[AsyncOpenAI]:
    async with (
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=demo_app),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as openai_client,
    ):
        yield openai_client


async def test_app_lists_exactly_the_documented_models(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert {model.id for model in response.data} == DOCUMENTED_MODEL_IDS
    extensions = {
        model.id: (model.model_extra or {})["lgos"] for model in response.data
    }
    # The UIs enable uploads, gateway MCP tools, background mode, and the
    # offered tools from these declarations alone; docs/demo/graphs/index.md
    # lists each model's features.
    assert {
        model_id: extension["features"]
        for model_id, extension in extensions.items()
        if extension["features"]
    } == {
        "background-interrupt": ["background", "interrupts"],
        "background-mock": ["background"],
        "deep-research": ["background", "file_inputs", "interrupts", "mcp_tools"],
        "file-input": ["file_inputs"],
        "interruptible-approval": ["interrupts"],
        "mcp-postgres": ["mcp_tools"],
        "nested-subgraphs": ["background", "interrupts"],
    }
    assert {
        model_id: (extension["server_tools"], extension["client_tools"])
        for model_id, extension in extensions.items()
        if extension["server_tools"] or extension["client_tools"]
    } == {
        "deep-research": (["web_search"], []),
        "persistent-plot-agent": ([], ["display_file"]),
        "server-tool": (["package_version", "web_search"], []),
    }


async def test_simple_model_retrieval_exposes_runtime_settings(
    openai_client: AsyncOpenAI,
) -> None:
    model = await openai_client.models.retrieve("simple-graph")

    extension = (model.model_extra or {})["lgos"]
    client_settings = extension["client_settings"]
    assert client_settings["defaults"] == {
        "use_history": False,
        "audience": "general",
        "model": settings.OPENAI_CHAT_COMPLETIONS_MODEL,
    }
    properties = client_settings["json_schema"]["properties"]
    assert properties["audience"]["enum"] == ["general", "beginner", "expert"]
    assert properties["model"]["enum"] == [
        settings.OPENAI_CHAT_COMPLETIONS_MODEL,
        settings.OPENAI_RESPONSES_MODEL,
    ]


async def test_simple_model_rejects_an_unoffered_model(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as error:
        await openai_client.responses.create(
            model="simple-graph",
            input="Hello",
            store=False,
            metadata={"lgos_settings": '{"model":"openai/unregistered"}'},
        )

    assert error.value.body["param"] == "metadata.lgos_settings"


@pytest.mark.parametrize(
    ("metadata", "expected_context"),
    [
        (None, SimpleContext()),
        (
            {"lgos_settings": '{"use_history":true}'},
            SimpleContext(use_history=True),
        ),
        (
            {"lgos_settings": '{"audience":"expert"}'},
            SimpleContext(audience="expert"),
        ),
        (
            {"lgos_settings": f'{{"model":"{settings.OPENAI_RESPONSES_MODEL}"}}'},
            SimpleContext(model=settings.OPENAI_RESPONSES_MODEL),
        ),
    ],
)
async def test_simple_model_builds_its_runtime_context(
    demo_app: FastAPI,
    metadata: dict[str, str] | None,
    expected_context: SimpleContext,
) -> None:
    graph_request = GraphRequest(
        model="simple-graph",
        metadata=metadata or {},
        user=None,
        tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
    )

    graph_config = demo_app.state.graph_registry.get_graph("simple-graph")
    graph = await graph_config.resolve_graph()

    assert await graph_config.build_context(graph_request, graph) == expected_context


async def test_custom_io_demo_works_through_openai_client(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.responses.create(
        store=False,
        model="custom-input-output-context",
        input=[{"role": "user", "content": "Show me custom schemas."}],
        safety_identifier="demo-user",
    )

    assert response.output_text == ("demo-user asked: Show me custom schemas.")


async def test_file_input_demo_prompts_for_an_attachment(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.responses.create(
        store=False,
        model="file-input",
        input=[{"role": "user", "content": "Summarize my file."}],
    )

    assert response.output_text == "Attach a file and try again."


async def test_response_outcomes_exposes_native_refusal(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.responses.create(
        store=False,
        model="response-outcomes",
        input="refusal",
    )

    assert response.status == "completed"
    assert response.output_text == ""  # ruff: ignore[compare-to-empty-string] - Distinguish the empty string from other falsey values.
    assert response.output[0].content[0].model_dump() == {
        "type": "refusal",
        "refusal": "I cannot help with bypassing safety controls.",
    }


async def test_response_outcomes_finishes_incomplete_stream_natively(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await openai_client.responses.create(
        store=False,
        model="response-outcomes",
        input="incomplete",
        stream=True,
    )
    events = [event async for event in stream]

    assert events[-1].type == "response.incomplete"
    assert not any(event.type == "response.completed" for event in events)
    response = events[-1].response
    assert response.status == "incomplete"
    assert response.completed_at is None
    assert response.incomplete_details.reason == "max_output_tokens"
    assert response.output_text == "This answer stopped before it could finish."
    assert response.output[0].status == "incomplete"


async def test_registry_compiles_graphs_with_server_resources(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    resources = ServerResources(
        checkpointer=sqlite_checkpointer,
        store=InMemoryStore(),
        run_coordinator=InMemoryRunCoordinator(),
    )
    async with open_registry(resources) as registry:
        for model_id in (
            "interruptible-approval",
            "background-interrupt",
            "nested-subgraphs",
        ):
            graph = await registry.get_graph(model_id).resolve_graph()
            assert graph.checkpointer is sqlite_checkpointer
        plot_agent = await registry.get_graph("persistent-plot-agent").resolve_graph()
        assert plot_agent.store is resources.store
        assert registry.run_coordinator is resources.run_coordinator


@pytest.mark.parametrize("tools", [[], [{"type": "custom", "name": "package_version"}]])
async def test_server_package_lookup_is_not_bound_when_unselected_or_disabled(
    demo_app: FastAPI,
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tools,
) -> None:
    class NoToolsModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            pytest.fail("Package lookup must not be bound when disabled.")

    model = NoToolsModel(responses=[AIMessage(content="Package lookup is disabled.")])
    monkeypatch.setattr("lgos_demo_api.utils.models.ChatOpenAI", lambda **kwargs: model)
    _rebuild_server_tool_graph(demo_app)
    details = await openai_client.models.retrieve("server-tool")
    assert "client_settings" not in details.lgos
    response = await openai_client.responses.create(
        model="server-tool",
        input="Which OpenAI SDK version is installed?",
        store=False,
        tools=tools,
        tool_choice="none" if tools else "auto",
    )
    assert response.output_text == "Package lookup is disabled."
    assert [item.type for item in response.output] == ["message"]


async def test_server_tool_graph_rejects_client_functions(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as error:
        await openai_client.responses.create(
            model="server-tool",
            input="Run this function.",
            tools=[{"type": "function", "name": "client_function"}],
        )

    assert error.value.response.json()["error"]["param"] == "tools"


@pytest.mark.parametrize(
    ("stream", "choice"),
    [
        (False, "auto"),
        (False, "required"),
        (True, "required"),
        (False, {"type": "custom", "name": "package_version"}),
    ],
)
async def test_server_custom_tool_executes_a_fresh_native_exchange(
    demo_app: FastAPI,
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    choice: str | dict[str, str],
) -> None:
    installed = f"openai=={version('openai')}"
    provider = ModelProvider(
        model_response(
            "Checking installed packages.",
            calls=[
                {
                    "id": "ctc_new",
                    "type": "custom_tool_call",
                    "call_id": "call_new",
                    "name": "package_version",
                    "input": "openai",
                    "status": "completed",
                }
            ],
        ),
        model_response(installed),
    )

    async with AsyncClient(transport=MockTransport(provider.handle)) as gateway:
        monkeypatch.setattr("lgos_demo_api.utils.models.GATEWAY_HTTP_CLIENT", gateway)
        _rebuild_server_tool_graph(demo_app)
        response = await openai_client.responses.create(
            model="server-tool",
            input=[
                {
                    "type": "custom_tool_call",
                    "call_id": "call_old",
                    "name": "package_version",
                    "input": "langgraph",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_old",
                    "output": "langgraph==previous-version",
                },
            ],
            tools=[{"type": "custom", "name": "package_version"}],
            tool_choice=choice,
            parallel_tool_calls=False,
            stream=stream,
        )
        if stream:
            events = [event async for event in response]
            response = events[-1].response
            assert (
                "".join(
                    event.delta
                    for event in events
                    if event.type == "response.output_text.delta"
                )
                == response.output_text
            )

    selecting, answering = provider.requests
    # The private selection call streams too, but its tokens stay hidden.
    assert [
        (body["store"], bool(body.get("stream"))) for body in provider.requests
    ] == [(False, stream)] * 2
    assert [(tool["type"], tool["name"]) for tool in selecting["tools"]] == [
        ("custom", "package_version")
    ]
    assert (selecting["parallel_tool_calls"], selecting["tool_choice"]) == (
        False,
        choice,
    )
    assert "tools" not in answering
    assert [
        item["output"]
        for item in answering["input"]
        if item.get("call_id") == "call_new"
        and item["type"] == "custom_tool_call_output"
    ] == [installed]

    assert response.status == "completed"
    output = [
        item for item in response.output if getattr(item, "phase", None) != "commentary"
    ]
    assert [item.type for item in output] == [
        "custom_tool_call",
        "custom_tool_call_output",
        "message",
    ]
    assert output[0].call_id == output[1].call_id == "call_new"
    assert (output[0].name, output[0].input) == ("package_version", "openai")
    assert output[1].output == output[-1].content[0].text == installed
    if stream:
        # The answer reaches the client in the provider's chunks, as it streams.
        assert [
            event.delta
            for event in events
            if event.type == "response.output_text.delta"
            and event.item_id == output[-1].id
        ] == text_chunks(installed)


@pytest.mark.parametrize("stream", [False, True])
async def test_server_web_search_runs_through_http_backend(
    demo_app: FastAPI,
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    model = SearchModel(
        expected_tools=[server_tool.web_search],
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "call_demo_search",
                        "name": "web_search",
                        "args": {"query": "OpenAI Responses API"},
                    }
                ],
            ),
            AIMessage(
                content=("See [OpenAI API](https://developers.openai.com/api/) docs."),
                response_metadata={"model_provider": "openai"},
            ),
        ],
    )

    async def search(_client, url, query):
        assert url == "https://searxng.example.com/search"
        assert query == "OpenAI Responses API"
        return [
            WebSearchResult.model_validate(
                {
                    "title": "OpenAI API",
                    "url": "https://developers.openai.com/api/",
                    "content": "Build with the OpenAI API.",
                }
            )
        ]

    monkeypatch.setattr(server_tool.settings, "WEB_SEARCH_BACKEND", "http")
    monkeypatch.setattr(
        server_tool.settings, "WEB_SEARCH_URL", "https://searxng.example.com/search"
    )
    monkeypatch.setattr("lgos_demo_api.utils.models.ChatOpenAI", lambda **kwargs: model)
    _rebuild_server_tool_graph(demo_app)
    monkeypatch.setattr(server_tool, "search_web", search)

    response = await openai_client.responses.create(
        model="server-tool",
        input="Find the Responses API documentation.",
        store=False,
        stream=stream,
        tools=[{"type": "web_search"}],
        tool_choice="required",
    )
    if stream:
        events = [event async for event in response]
        response = events[-1].response
        assert [
            event.item.type
            for event in events
            if event.type == "response.output_item.done"
            and getattr(event.item, "phase", None) != "commentary"
        ] == ["web_search_call", "message"]

    output = [
        item for item in response.output if getattr(item, "phase", None) != "commentary"
    ]
    assert [item.type for item in output] == ["web_search_call", "message"]
    assert output[0].action.query == "OpenAI Responses API"
    assert output[-1].content[0].text == (
        "See [OpenAI API](https://developers.openai.com/api/) docs."
    )
    assert output[1].content[0].annotations[0].url == (
        "https://developers.openai.com/api/"
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_server_web_search_can_use_the_upstream_openai_tool(
    demo_app: FastAPI,
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    make_tool_calling_model,
    stream: bool,
) -> None:
    provider = SearchModel(
        expected_tools=[{"type": "web_search"}],
        responses=[
            AIMessage(
                content=[
                    {
                        "type": "server_tool_call",
                        "name": "web_search",
                        "id": "ws_demo_provider",
                        "args": {
                            "type": "search",
                            "query": "OpenAI Responses API",
                        },
                    },
                    {
                        "type": "server_tool_result",
                        "tool_call_id": "ws_demo_provider",
                        "status": "success",
                    },
                    {
                        "type": "text",
                        "text": "Private search summary",
                        "annotations": [
                            {
                                "type": "citation",
                                "url": "https://developers.openai.com/api/",
                                "title": "OpenAI API",
                                "start_index": 0,
                                "end_index": 6,
                            },
                        ],
                    },
                ]
            )
        ],
    )

    model = make_tool_calling_model(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_demo_search",
                    "name": "web_search",
                    "args": {"query": "OpenAI Responses API"},
                }
            ],
        ),
        AIMessage(content="See [OpenAI API](https://developers.openai.com/api/)."),
    )
    models = iter([model, model, provider])
    monkeypatch.setattr(server_tool.settings, "WEB_SEARCH_BACKEND", "openai")
    monkeypatch.setattr(
        "lgos_demo_api.utils.models.ChatOpenAI", lambda **kwargs: next(models)
    )
    _rebuild_server_tool_graph(demo_app)

    response = await openai_client.responses.create(
        model="server-tool",
        input="Find the Responses API documentation.",
        store=False,
        stream=stream,
        tools=[{"type": "web_search"}],
        tool_choice="required",
    )
    if stream:
        events = [event async for event in response]
        response = events[-1].response

    output = [
        item for item in response.output if getattr(item, "phase", None) != "commentary"
    ]
    assert [item.type for item in output] == ["web_search_call", "message"]
    assert output[0].action.query == "OpenAI Responses API"
    assert (
        output[-1].content[0].text
        == "See [OpenAI API](https://developers.openai.com/api/)."
    )
    assert (
        output[-1].content[0].annotations[0].url == "https://developers.openai.com/api/"
    )
