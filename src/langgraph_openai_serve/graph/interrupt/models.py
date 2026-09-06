"""Protocol-neutral models for interrupt-enabled graph runs."""

from dataclasses import dataclass

from langgraph.types import Interrupt


@dataclass(frozen=True, slots=True)
class InterruptResume:
    """A complete, causally bound set of interrupt answers."""

    run_id: str
    state_token: str
    values: dict[str, str]


@dataclass(frozen=True)
class LangGraphInterruptBatch:
    """The durable interrupts awaiting answers for one graph run."""

    run_id: str
    state_token: str
    interrupts: tuple[Interrupt, ...]


__all__ = ["InterruptResume", "LangGraphInterruptBatch"]
