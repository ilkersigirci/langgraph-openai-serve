"""Strict codec for durable Responses interrupt continuations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lgos_chainlit.lgos_protocol import INTERRUPT_TOOL_NAME

INTERRUPT_LEDGER_METADATA_KEY = "lgos_chainlit.hitl_interrupt_ledger"
INTERRUPT_LEDGER_SCHEMA_VERSION = 2
PENDING_LEDGER_STATUS = "pending"
COMPLETED_LEDGER_STATUS = "completed"


class InvalidInterruptLedgerError(ValueError):
    """A persisted Chainlit interrupt ledger is unsafe to resume."""


@dataclass(frozen=True, slots=True)
class InterruptContinuation:
    """The validated Responses values needed to resume one complete batch."""

    model_id: str
    response_id: str
    function_calls: tuple[ResponseFunctionToolCall, ...]


@dataclass(frozen=True, slots=True)
class PendingLedgerEntry:
    """The host step and decoded value for the newest pending ledger."""

    step: Mapping[str, object]
    continuation: InterruptContinuation


class _LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _PersistedInterruptCall(_LedgerModel):
    arguments: str
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: Literal["function_call"]
    id: str | None = Field(default=None, min_length=1)
    caller: None = None
    namespace: None = None
    status: Literal["completed"] | None = None


class _PendingLedger(_LedgerModel):
    schema_version: Literal[2]
    status: Literal["pending"]
    model_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    function_calls: list[_PersistedInterruptCall] = Field(min_length=1)


class _CompletedLedger(_LedgerModel):
    schema_version: Literal[2]
    status: Literal["completed"]


def interrupt_continuation(
    *,
    model_id: str,
    response_id: str,
    function_calls: Sequence[ResponseFunctionToolCall],
) -> InterruptContinuation:
    """Validate SDK output before it becomes client-owned durable state."""
    raw_ledger = {
        "schema_version": INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": PENDING_LEDGER_STATUS,
        "model_id": model_id,
        "response_id": response_id,
        "function_calls": [
            call.model_dump(mode="json", exclude_none=True) for call in function_calls
        ],
    }
    return _decode_pending(raw_ledger)


def pending_ledger_metadata(
    continuation: InterruptContinuation,
) -> dict[str, object]:
    """Encode a pending continuation for public Chainlit message metadata."""
    ledger = _validate_pending(
        {
            "schema_version": INTERRUPT_LEDGER_SCHEMA_VERSION,
            "status": PENDING_LEDGER_STATUS,
            "model_id": continuation.model_id,
            "response_id": continuation.response_id,
            "function_calls": [
                call.model_dump(mode="json", exclude_none=True)
                for call in continuation.function_calls
            ],
        }
    )
    return ledger.model_dump(mode="json", exclude_none=True)


def completed_ledger_metadata() -> dict[str, object]:
    """Encode the terminal marker that prevents older pending replay."""
    ledger = _CompletedLedger(
        schema_version=INTERRUPT_LEDGER_SCHEMA_VERSION,
        status=COMPLETED_LEDGER_STATUS,
    )
    return ledger.model_dump(mode="json")


def decode_interrupt_ledger(raw_ledger: object) -> InterruptContinuation | None:
    """Decode one strict ledger value; completed values deliberately return none."""
    if not isinstance(raw_ledger, dict):
        msg = "Interrupt ledger metadata is not an object."
        raise InvalidInterruptLedgerError(msg)
    if raw_ledger.get("schema_version") != INTERRUPT_LEDGER_SCHEMA_VERSION:
        msg = "Interrupt ledger schema is unsupported."
        raise InvalidInterruptLedgerError(msg)

    status = raw_ledger.get("status")
    if status == COMPLETED_LEDGER_STATUS:
        try:
            _CompletedLedger.model_validate(raw_ledger)
        except ValidationError as exc:
            msg = "Completed interrupt ledger is invalid."
            raise InvalidInterruptLedgerError(msg) from exc
        return None
    if status != PENDING_LEDGER_STATUS:
        msg = "Interrupt ledger status is invalid."
        raise InvalidInterruptLedgerError(msg)
    return _decode_pending(raw_ledger)


def newest_pending_ledger(steps: object) -> PendingLedgerEntry | None:
    """Select the newest ledger step, with completion blocking older replay."""
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        msg = "Persisted Chainlit steps are invalid."
        raise InvalidInterruptLedgerError(msg)

    for step in reversed(steps):
        if not isinstance(step, Mapping):
            continue
        metadata = step.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or INTERRUPT_LEDGER_METADATA_KEY not in metadata
        ):
            continue

        continuation = decode_interrupt_ledger(metadata[INTERRUPT_LEDGER_METADATA_KEY])
        if continuation is None:
            return None
        return PendingLedgerEntry(step=step, continuation=continuation)
    return None


def _decode_pending(raw_ledger: object) -> InterruptContinuation:
    ledger = _validate_pending(raw_ledger)
    calls = tuple(
        ResponseFunctionToolCall.model_validate(
            call.model_dump(mode="python", exclude_none=True)
        )
        for call in ledger.function_calls
    )
    if any(call.name != INTERRUPT_TOOL_NAME for call in calls):
        msg = "Interrupt ledger has no valid interrupt calls."
        raise InvalidInterruptLedgerError(msg)
    call_ids = {call.call_id for call in calls}
    if len(call_ids) != len(calls):
        msg = "Interrupt ledger call IDs must be unique."
        raise InvalidInterruptLedgerError(msg)
    return InterruptContinuation(
        model_id=ledger.model_id,
        response_id=ledger.response_id,
        function_calls=calls,
    )


def _validate_pending(raw_ledger: object) -> _PendingLedger:
    try:
        return _PendingLedger.model_validate(raw_ledger)
    except ValidationError as exc:
        msg = "Interrupt ledger continuation is invalid."
        raise InvalidInterruptLedgerError(msg) from exc
