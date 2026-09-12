from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph_openai_serve import GraphRequest
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from openai import AsyncOpenAI, BadRequestError

from lgos_demo_api import app as app_module
from lgos_demo_api.checkpointer import PostgresRuntime
from lgos_demo_api.graphs import hosted_tool
from lgos_demo_api.graphs.simple import SimpleContext
from lgos_demo_api.utils.web_search import WebSearchResult

DOCUMENTED_MODEL_IDS = {
    "advanced-mcp-tools",
    "citation-events",
    "complex-subgraphs",
    "custom-event-showcase",
    "custom-input-output-context",
    "file-input",
    "interruptible-approval",
    "lgos-rag",
    "persistent-plot-agent",
    "multi-node-streaming",
    "simple-graph",
    "hosted-tool",
    "simple-graph-external-tools",
    "status-events",
}
CLIENT_SETTINGS_SCHEMA_VERSION = 1


def _rebuild_hosted_tool_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hosted_tool.hosted_tool_graph_config,
        "graph",
        hosted_tool.create_hosted_tool_graph(),
    )


@pytest.fixture
def demo_app() -> FastAPI:
    return app_module.create_custom_app()


@pytest.fixture
async def openai_client(demo_app: FastAPI) -> AsyncIterator[AsyncOpenAI]:
    async with (
        AsyncClient(
            transport=ASGITransport(app=demo_app),
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

    assert response.object == "list"
    assert {model.id for model in response.data} == DOCUMENTED_MODEL_IDS
    descriptions = {
        model.id: (model.model_extra or {})["lgos"]["description"]
        for model in response.data
    }
    assert all(description.strip() for description in descriptions.values())
    features = {
        model.id: (model.model_extra or {})["lgos"]["features"]
        for model in response.data
    }
    assert features["file-input"] == ["file_inputs"]

    interrupt_model = await openai_client.models.retrieve("interruptible-approval")
    extension = (interrupt_model.model_extra or {})["lgos"]
    assert extension == {
        "schema_version": 1,
        "description": descriptions["interruptible-approval"],
        "features": ["interrupts"],
    }

    for model_id in ("complex-subgraphs", "custom-event-showcase", "status-events"):
        model = await openai_client.models.retrieve(model_id)
        extension = (model.model_extra or {})["lgos"]
        assert extension == {
            "schema_version": 1,
            "description": descriptions[model_id],
            "features": ["client_events"],
        }

    plot_model = await openai_client.models.retrieve("persistent-plot-agent")
    plot_extension = (plot_model.model_extra or {})["lgos"]
    assert plot_extension["features"] == []
    assert plot_extension["client_settings"]["defaults"] == {
        "chart_type": "bar",
        "currency": "USD",
        "show_legend": True,
    }
    assert plot_extension["client_settings"]["json_schema"]["properties"]["chart_type"][
        "enum"
    ] == ["bar", "line"]


async def test_cors_exposes_request_id(demo_app: FastAPI) -> None:
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/v1/health",
            headers={"Origin": "https://client.example"},
        )

    assert response.headers["access-control-expose-headers"] == "X-Request-ID"
    assert response.headers["x-request-id"]


async def test_simple_model_retrieval_exposes_runtime_settings(
    openai_client: AsyncOpenAI,
) -> None:
    model = await openai_client.models.retrieve("simple-graph")

    extension = (model.model_extra or {})["lgos"]
    client_settings = extension["client_settings"]
    assert client_settings["schema_version"] == CLIENT_SETTINGS_SCHEMA_VERSION
    assert client_settings["defaults"] == {
        "use_history": False,
        "audience": "general",
    }
    assert client_settings["json_schema"]["properties"]["audience"]["enum"] == [
        "general",
        "beginner",
        "expert",
    ]


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
        user="demo-user",
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


async def test_complex_subgraphs_preserve_streaming_parity(
    openai_client: AsyncOpenAI,
) -> None:
    complete = await openai_client.responses.create(
        store=False,
        model="complex-subgraphs",
        input=[{"role": "user", "content": "Show nested subgraph routing docs."}],
    )
    stream = await openai_client.responses.create(
        store=False,
        model="complex-subgraphs",
        input=[{"role": "user", "content": "Show nested subgraph routing docs."}],
        stream=True,
    )

    phases = {}
    final_deltas = []
    async for event in stream:
        if event.type == "response.output_item.added" and event.item.type == "message":
            phases[event.output_index] = event.item.phase
        elif (
            event.type == "response.output_text.delta"
            and phases.get(event.output_index) == "final_answer"
        ):
            final_deltas.append(event.delta)
    streamed = "".join(final_deltas)

    assert streamed == complete.output_text


async def test_lifespan_installs_shared_postgres_runtime(
    demo_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    coordinator = InMemoryRunCoordinator()

    runtime = PostgresRuntime(
        checkpointer=sqlite_checkpointer,  # type: ignore[arg-type]
        store=InMemoryStore(),  # type: ignore[arg-type]
        run_coordinator=coordinator,  # type: ignore[arg-type]
    )

    @asynccontextmanager
    async def postgres_runtime(postgres_uri: str):
        assert postgres_uri == app_module.settings.POSTGRES_URI
        yield runtime

    runtime_factory = Mock(wraps=postgres_runtime)
    monkeypatch.setattr(app_module, "postgres_runtime", runtime_factory)

    async with app_module.lifespan(demo_app):
        assert demo_app.state.interruptible_graph.checkpointer is sqlite_checkpointer
        assert demo_app.state.run_coordinator is coordinator
        assert demo_app.state.persistent_plot_agent.store is runtime.store

        config = demo_app.state.graph_registry.get_graph("interruptible-approval")
        assert config.run_coordinator is not None
        async with config.run_coordinator("thread-1"):
            pass

    runtime_factory.assert_called_once_with(app_module.settings.POSTGRES_URI)


def test_main_leaves_access_logging_to_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)

    app_module.main()

    run.assert_called_once()
    assert run.call_args.kwargs["access_log"] is False


@pytest.mark.parametrize(
    "tools", [[], [{"type": "custom", "name": "lgos_current_time"}]]
)
async def test_hosted_time_lookup_is_not_bound_when_unselected_or_disabled(
    openai_client: AsyncOpenAI, monkeypatch: pytest.MonkeyPatch, tools
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    class NoToolsModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            pytest.fail("Time lookup must not be bound when disabled.")

    model = NoToolsModel(responses=[AIMessage(content="Time lookup is disabled.")])
    monkeypatch.setattr(hosted_tool, "ChatOpenAI", lambda **kwargs: model)
    _rebuild_hosted_tool_graph(monkeypatch)
    details = await openai_client.models.retrieve("hosted-tool")
    assert "client_settings" not in details.lgos
    assert "hosted_tools" not in details.lgos
    response = await openai_client.responses.create(
        model="hosted-tool",
        input="What time is it?",
        store=False,
        tools=tools,
        tool_choice="none" if tools else "auto",
    )
    assert response.output_text == "Time lookup is disabled."
    assert [item.type for item in response.output] == ["message"]


async def test_hosted_tool_rejects_client_functions(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as error:
        await openai_client.responses.create(
            model="hosted-tool",
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
        (False, {"type": "custom", "name": "lgos_current_time"}),
    ],
)
async def test_hosted_tool_executes_a_fresh_call_and_returns_only_the_final_answer(
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    choice: str | dict[str, str],
) -> None:
    import json

    from httpx import MockTransport, Request, Response
    from langchain_openai import ChatOpenAI

    requests = []

    def respond(request: Request) -> Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path.endswith("/responses")
        # Server-tool requests use complete model turns, not token streaming.
        assert not body.get("stream")
        assert body["store"] is False
        assert body["tools"][0]["type"] == "custom"
        assert body["tools"][0]["name"] == "lgos_current_time"
        assert body["parallel_tool_calls"] is False
        first_call = len(requests) == 1
        assert body["tool_choice"] == (choice if first_call else "auto")
        if not first_call:
            result = body["input"][-1]
            assert result["type"] == "custom_tool_call_output"
            assert result["call_id"] == "call_new"
            assert result["output"].startswith("Europe/Istanbul: ")
            assert result["output"].endswith("+03:00")
        output = [
            {
                "id": f"msg_{len(requests)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Checking the clock."
                        if first_call
                        else result["output"],
                        "annotations": [],
                    }
                ],
            }
        ]
        if first_call:
            output.append(
                {
                    "id": "ctc_new",
                    "type": "custom_tool_call",
                    "call_id": "call_new",
                    "name": "lgos_current_time",
                    "input": "Europe/Istanbul",
                    "status": "completed",
                }
            )
        return Response(
            200,
            json={
                "id": f"resp_{len(requests)}",
                "object": "response",
                "created_at": 1,
                "model": "test-model",
                "status": "completed",
                "output": output,
            },
        )

    async with AsyncClient(transport=MockTransport(respond)) as provider:
        monkeypatch.setattr(
            hosted_tool,
            "ChatOpenAI",
            lambda **kwargs: ChatOpenAI(http_async_client=provider, **kwargs),
        )
        _rebuild_hosted_tool_graph(monkeypatch)
        response = await openai_client.responses.create(
            model="hosted-tool",
            input=[
                {
                    "type": "custom_tool_call",
                    "call_id": "call_old",
                    "name": "lgos_current_time",
                    "input": "Europe/Istanbul",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_old",
                    "output": "An old timestamp",
                },
            ],
            tools=[{"type": "custom", "name": "lgos_current_time"}],
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

    assert len(requests) == 2
    assert response.status == "completed"
    assert [item.type for item in response.output] == [
        "custom_tool_call",
        "custom_tool_call_output",
        "message",
    ]
    assert response.output[0].call_id == response.output[1].call_id == "call_new"
    assert response.output[0].name == "lgos_current_time"
    assert response.output[0].input == "Europe/Istanbul"
    assert response.output[1].output == response.output_text


@pytest.mark.parametrize("stream", [False, True])
async def test_hosted_web_search_runs_through_http_backend(
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    class SearchModel(FakeMessagesListChatModel):
        calls: int = 0

        def bind_tools(self, tools, **kwargs):
            self.calls += 1
            assert tools == [hosted_tool.web_search]
            assert kwargs["tool_choice"] == ("required" if self.calls == 1 else "auto")
            return self

    model = SearchModel(
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
        ]
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

    monkeypatch.setattr(hosted_tool, "ChatOpenAI", lambda **kwargs: model)
    _rebuild_hosted_tool_graph(monkeypatch)
    monkeypatch.setattr(hosted_tool, "search_web", search)

    response = await openai_client.responses.create(
        model="hosted-tool",
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
        ] == ["web_search_call", "message"]
        assert [
            event.delta
            for event in events
            if event.type == "response.output_text.delta"
        ] == [response.output_text]

    assert [item.type for item in response.output] == ["web_search_call", "message"]
    assert response.output[0].action.query == "OpenAI Responses API"
    assert response.output_text == (
        "See [OpenAI API](https://developers.openai.com/api/) docs."
    )
    assert response.output[1].content[0].annotations[0].url == (
        "https://developers.openai.com/api/"
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_hosted_web_search_can_use_the_upstream_openai_tool(
    openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    class SearchModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert tools == [{"type": "web_search"}]
            assert kwargs["tool_choice"] == "required"
            return self

    model = SearchModel(
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
                    {"type": "text", "text": "OpenAI docs"},
                ]
            )
        ]
    )

    monkeypatch.setattr(hosted_tool.settings, "WEB_SEARCH_BACKEND", "openai")
    monkeypatch.setattr(hosted_tool, "ChatOpenAI", lambda **kwargs: model)
    _rebuild_hosted_tool_graph(monkeypatch)

    response = await openai_client.responses.create(
        model="hosted-tool",
        input="Find the Responses API documentation.",
        store=False,
        stream=stream,
        tools=[{"type": "web_search"}],
        tool_choice="required",
    )
    if stream:
        events = [event async for event in response]
        response = events[-1].response

    assert [item.type for item in response.output] == ["web_search_call", "message"]
    assert response.output[0].id == "ws_demo_provider"
    assert response.output[0].action.query == "OpenAI Responses API"
    assert response.output_text == "OpenAI docs"
