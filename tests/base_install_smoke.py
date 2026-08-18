"""Exercise the installed wheel without optional or test dependencies."""

import asyncio
import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def assert_optional_integrations_are_not_installed() -> None:
    for module in (
        "langchain",
        "langfuse",
        "langgraph.checkpoint.postgres",
        "psycopg",
        "psycopg_pool",
    ):
        assert importlib.util.find_spec(module) is None, (
            f"Optional integration dependency unexpectedly installed: {module}"
        )


@contextmanager
def hostile_working_directory() -> Iterator[None]:
    original_working_directory = Path.cwd()
    with TemporaryDirectory() as temporary_directory:
        Path(temporary_directory, ".env").write_text(
            "LGOS_OPENAI_API_PREFIX=not-a-path\nLGOS_OPENAI_API_DOCS_ENABLED=not-a-boolean\nLGOS_ENABLE_LANGFUSE=true",
            encoding="utf-8",
        )
        try:
            os.chdir(temporary_directory)
            yield
        finally:
            os.chdir(original_working_directory)


async def main() -> None:
    assert_optional_integrations_are_not_installed()

    # Import only after proving no optional integration was installed, and from
    # a directory whose dotenv values must not configure the library.
    with hostile_working_directory():
        from httpx import (
            ASGITransport,
            AsyncClient,
        )
        from langchain_core.messages import (
            AIMessage,
        )
        from langgraph.graph import (
            MessagesState,
            StateGraph,
        )
        from openai import AsyncOpenAI

        from langgraph_openai_serve import (
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
