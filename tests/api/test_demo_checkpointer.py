from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Self

import pytest
from demo.api import checkpointer as demo_checkpointer


@pytest.mark.anyio
async def test_postgres_checkpointer_owns_process_local_pool(monkeypatch) -> None:
    events: list[str] = []

    class FakePool(AbstractAsyncContextManager):
        def __init__(self, **kwargs) -> None:
            assert kwargs == {
                "conninfo": "postgresql://example",
                "kwargs": {
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": demo_checkpointer.dict_row,
                },
                "min_size": 1,
                "max_size": 5,
                "open": False,
            }

        async def __aenter__(self) -> Self:
            events.append("pool opened")
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            events.append("pool closed")

    class FakeSaver:
        def __init__(self, pool: FakePool) -> None:
            events.append("saver created")

        async def setup(self) -> None:
            events.append("schema set up")

    monkeypatch.setattr(demo_checkpointer, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(demo_checkpointer, "AsyncPostgresSaver", FakeSaver)

    async with demo_checkpointer.postgres_checkpointer(
        "postgresql://example"
    ) as checkpointer:
        assert isinstance(checkpointer, FakeSaver)
        assert events == ["pool opened", "saver created"]

    assert events[-1] == "pool closed"


@pytest.mark.anyio
async def test_setup_postgres_schema_runs_setup_explicitly(monkeypatch) -> None:
    setup_calls = 0

    class FakeSaver:
        async def setup(self) -> None:
            nonlocal setup_calls
            setup_calls += 1

    class FakeCheckpointerContext:
        async def __aenter__(self) -> FakeSaver:
            return FakeSaver()

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

    def fake_postgres_checkpointer(postgres_uri: str) -> FakeCheckpointerContext:
        assert postgres_uri == "postgresql://example"
        return FakeCheckpointerContext()

    monkeypatch.setattr(
        demo_checkpointer,
        "postgres_checkpointer",
        fake_postgres_checkpointer,
    )

    await demo_checkpointer.setup_postgres_schema("postgresql://example")

    assert setup_calls == 1
