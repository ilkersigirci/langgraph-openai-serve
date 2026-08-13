"""Result models produced by interrupt-enabled graph runs."""

from dataclasses import dataclass

from langgraph.types import Interrupt


@dataclass(frozen=True)
class LangGraphInterruptBatch:
    """The durable interrupts awaiting answers for one graph run."""

    run_id: str
    state_token: str
    interrupts: tuple[Interrupt, ...]


__all__ = ["LangGraphInterruptBatch"]
