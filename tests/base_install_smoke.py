"""Exercise the installed wheel without optional or test dependencies."""

import asyncio
import importlib.util


def assert_postgres_is_not_installed() -> None:
    for module in (
        "langgraph.checkpoint.postgres",
        "psycopg",
        "psycopg_pool",
    ):
        assert importlib.util.find_spec(module) is None, (
            f"PostgreSQL dependency unexpectedly installed: {module}"
        )


async def main() -> None:
    assert_postgres_is_not_installed()

    # Import only after proving no transitive dependency installed PostgreSQL.
    from httpx import ASGITransport, AsyncClient  # noqa: PLC0415
    from langchain_core.messages import AIMessage  # noqa: PLC0415
    from langgraph.graph import MessagesState, StateGraph  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    from langgraph_openai_serve import (  # noqa: PLC0415
        GraphConfig,
        GraphRegistry,
        LanggraphOpenaiServe,
    )

    async def respond(_state: MessagesState):
        return {"messages": [AIMessage(content="base install works")]}

    graph = (
        StateGraph(MessagesState)
        .add_node("respond", respond)
        .set_entry_point("respond")
        .set_finish_point("respond")
        .compile()
    )
    app = (
        LanggraphOpenaiServe(
            graphs=GraphRegistry(
                registry={
                    "minimal": GraphConfig(
                        graph=graph,
                        description="Minimal graph",
                        streamable_node_names=["respond"],
                    )
                }
            )
        )
        .bind_openai_api()
        .app
    )

    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as openai_client,
    ):
        response = await openai_client.chat.completions.create(
            model="minimal",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert response.choices[0].message.content == "base install works"


if __name__ == "__main__":
    asyncio.run(main())
