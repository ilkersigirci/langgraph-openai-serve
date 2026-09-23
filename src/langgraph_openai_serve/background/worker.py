"""Checkpoint-aware graph execution for native background workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from openai.types.responses import Response
from pydantic import ValidationError

from langgraph_openai_serve.api.responses.output import (
    UnsupportedResponsesOutputError,
)
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_responses_request,
    selected_server_tools,
    validate_tools,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background.contracts import BackgroundSettings
from langgraph_openai_serve.background.responses import (
    failed_response,
    finish_response,
    output_response,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.coordination import RunBusyError
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphNotFoundError,
)
from langgraph_openai_serve.graph.runner import (
    BackgroundCheckpointIncompleteError,
    BackgroundGraphInterruptedError,
    run_background_graph,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import BaseMessage

    from langgraph_openai_serve.background.store import ResponseStore, StoredRun
    from langgraph_openai_serve.graph.coordination import RunLease
    from langgraph_openai_serve.graph.graph_registry import GraphRegistry
    from langgraph_openai_serve.graph.request import GraphRequest


class _GraphVersionMismatchError(RuntimeError):
    """The registered graph cannot resume a job persisted for another version."""


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    request: ResponseCreateRequest
    graph_config: GraphConfig
    graph_request: GraphRequest
    messages: list[BaseMessage]


# Retrying cannot fix these, so they become a failed Response. Every other
# exception propagates and the background engine retries the delivery.
_PERMANENT_ERRORS = (
    BackgroundCheckpointIncompleteError,
    BackgroundGraphInterruptedError,
    GraphConfigurationError,
    GraphNotFoundError,
    UnsupportedResponsesOutputError,
    UnsupportedResponsesRequestError,
    ValidationError,
    _GraphVersionMismatchError,
)


class BackgroundWorker:
    """Run persisted work while the background engine owns delivery and retries."""

    def __init__(
        self,
        *,
        graphs: GraphRegistry,
        store: ResponseStore,
        settings: BackgroundSettings | None = None,
    ) -> None:
        self.graphs = graphs
        self.store = store
        self.settings = settings or BackgroundSettings()

    async def execute(self, response_id: str) -> None:
        """Execute one engine delivery, resuming from LangGraph checkpoints."""
        await self._deliver(response_id, finalize_only=False)

    async def finalize(self, response_id: str) -> None:
        """Publish checkpointed output or failure after the engine exhausts retries."""
        await self._deliver(response_id, finalize_only=True)

    async def maintain(
        self,
        *,
        resubmit: Callable[[StoredRun], Awaitable[None]] | None = None,
    ) -> dict[str, int]:
        """
        Run bounded upkeep on the engine's recurring schedule.

        ``resubmit`` submits a run again. With it, runs queued longer than
        ``resubmit_after`` are resubmitted, which recovers a failed submission
        or an API crash between persisting a run and submitting it.
        """
        resubmitted = await self._resubmit_queued(resubmit) if resubmit else 0
        cleaned = await self._clean_pending()
        expired = await self.store.expire(
            now=datetime.now(UTC),
            limit=self.settings.maintenance_batch_size,
        )
        return {"resubmitted": resubmitted, "cleaned": cleaned, "expired": expired}

    async def _deliver(self, response_id: str, *, finalize_only: bool) -> None:
        run = await self.store.get(response_id)
        if run is None:
            return
        if run.terminal:
            await self._cleanup(run)
            return
        try:
            prepared = self._prepare(run)
        except _PERMANENT_ERRORS as exc:
            terminal = await self._fail(run, *_public_failure(exc))
            if terminal is not None:
                await self._cleanup(terminal)
            return

        coordinator = prepared.graph_config.run_coordinator
        if coordinator is None:  # GraphConfig requires one for background graphs.
            msg = "Background graph has no run coordinator."
            raise RuntimeError(msg)
        async with coordinator(run.checkpoint_thread_id) as lease:
            await self._run_locked(run, prepared, lease, finalize_only=finalize_only)

    def _prepare(self, run: StoredRun) -> _PreparedRun:
        request = ResponseCreateRequest.model_validate(run.envelope)
        graph_config = self.graphs.get_graph(run.model)
        if graph_config.background_version != run.graph_version:
            msg = "The registered background graph version is incompatible."
            raise _GraphVersionMismatchError(msg)
        validate_tools(request, graph_config.server_tools)
        graph_request, messages, _ = decode_responses_request(
            request,
            graph_config.server_tools,
        )
        return _PreparedRun(request, graph_config, graph_request, messages)

    async def _run_locked(
        self,
        run: StoredRun,
        prepared: _PreparedRun,
        lease: RunLease,
        *,
        finalize_only: bool,
    ) -> None:
        # Re-read under the lease: another delivery may have finished the run.
        current = await self.store.get(run.response_id)
        if current is not None and not current.terminal and not finalize_only:
            current = await self.store.mark_in_progress(
                run.response_id,
                now=datetime.now(UTC),
            )
        if current is None:
            return
        if current.terminal:
            await self._delete_checkpoints(current, prepared.graph_config, lease)
            return

        try:
            response = await self._render(
                current, prepared, finalize_only=finalize_only
            )
        except _PERMANENT_ERRORS as exc:
            response = self._failure_response(current, *_public_failure(exc))
        lease.ensure_owned()
        winner = await finish_response(
            self.store,
            self.settings,
            current.response_id,
            response,
        )
        if winner is not None:
            await self._delete_checkpoints(winner, prepared.graph_config, lease)

    @staticmethod
    async def _render(
        run: StoredRun,
        prepared: _PreparedRun,
        *,
        finalize_only: bool,
    ) -> Response:
        result = await run_background_graph(
            prepared.graph_request,
            prepared.messages,
            prepared.graph_config,
            checkpoint_thread_id=run.checkpoint_thread_id,
            finalize_only=finalize_only,
        )
        server_tools = selected_server_tools(
            prepared.request,
            prepared.graph_config.server_tools,
        )
        if server_tools and not result.root_messages:
            msg = (
                "Background server-tool output must retain the root messages "
                "transcript until publication."
            )
            raise UnsupportedResponsesOutputError(msg)
        return output_response(
            prepared.request,
            result.message,
            response_id=run.response_id,
            created_at=int(run.created_at.timestamp()),
            server_tools=server_tools,
            root_messages=result.root_messages,
            initial_call_ids=frozenset(run.initial_call_ids),
        )

    @staticmethod
    def _failure_response(run: StoredRun, message: str, code: str) -> Response:
        logger.warning(
            "background.response_failed",
            extra={"response_id": run.response_id, "failure_code": code},
        )
        return failed_response(Response.model_validate(run.response), message=message)

    async def _fail(self, run: StoredRun, message: str, code: str) -> StoredRun | None:
        return await finish_response(
            self.store,
            self.settings,
            run.response_id,
            self._failure_response(run, message, code),
        )

    async def _resubmit_queued(
        self,
        resubmit: Callable[[StoredRun], Awaitable[None]],
    ) -> int:
        now = datetime.now(UTC)
        runs = await self.store.claim_queued(
            created_before=now - self.settings.resubmit_after,
            now=now,
            limit=self.settings.maintenance_batch_size,
        )
        resubmitted = 0
        for run in runs:
            try:
                await resubmit(run)
            except Exception:
                logger.exception(
                    "background.resubmission_failed",
                    extra={"response_id": run.response_id},
                )
                continue
            resubmitted += 1
        return resubmitted

    async def _clean_pending(self) -> int:
        runs = await self.store.claim_cleanup_ready(
            now=datetime.now(UTC),
            limit=self.settings.maintenance_batch_size,
        )
        cleaned = 0
        for run in runs:
            try:
                cleaned += int(await self._cleanup(run))
            except RunBusyError:
                continue
            except Exception:
                logger.exception(
                    "background.checkpoint_cleanup_failed",
                    extra={"response_id": run.response_id},
                )
        return cleaned

    async def _cleanup(self, run: StoredRun) -> bool:
        if not run.cleanup_pending:
            return False
        try:
            graph_config = self.graphs.get_graph(run.model)
        except GraphNotFoundError:
            return await self._abandon_cleanup(run)
        coordinator = graph_config.run_coordinator
        if coordinator is None:
            return await self._abandon_cleanup(run)
        async with coordinator(run.checkpoint_thread_id) as lease:
            current = await self.store.get(run.response_id)
            if current is None or not current.cleanup_pending:
                return False
            return await self._delete_checkpoints(current, graph_config, lease)

    async def _delete_checkpoints(
        self,
        run: StoredRun,
        graph_config: GraphConfig,
        lease: RunLease,
    ) -> bool:
        try:
            graph = await graph_config.resolve_graph()
        except GraphConfigurationError:
            return await self._abandon_cleanup(run)
        checkpointer = graph.checkpointer
        if not isinstance(checkpointer, BaseCheckpointSaver):
            return await self._abandon_cleanup(run)
        lease.ensure_owned()
        await checkpointer.adelete_thread(run.checkpoint_thread_id)
        await self.store.finish_cleanup(run.response_id, now=datetime.now(UTC))
        return True

    async def _abandon_cleanup(self, run: StoredRun) -> bool:
        logger.warning(
            "background.checkpoint_cleanup_abandoned",
            extra={"response_id": run.response_id, "model": run.model},
        )
        await self.store.finish_cleanup(run.response_id, now=datetime.now(UTC))
        return False


_FAILURE_DETAILS: tuple[tuple[type[BaseException], str, str], ...] = (
    (
        BackgroundCheckpointIncompleteError,
        "Background execution ended before producing complete output.",
        "background_execution_failed",
    ),
    (
        BackgroundGraphInterruptedError,
        "Background responses do not support graph interrupts.",
        "background_interrupt_unsupported",
    ),
    (
        UnsupportedResponsesOutputError,
        "Background graph output cannot be reconstructed safely.",
        "background_output_unsupported",
    ),
    (
        UnsupportedResponsesRequestError,
        "The persisted background request is not supported.",
        "background_request_unsupported",
    ),
    (
        GraphNotFoundError,
        "The background model is no longer registered.",
        "background_configuration_error",
    ),
    (
        _GraphVersionMismatchError,
        "The registered background graph version is incompatible with this job.",
        "background_graph_incompatible",
    ),
)


def _public_failure(exc: BaseException) -> tuple[str, str]:
    for error_type, message, code in _FAILURE_DETAILS:
        if isinstance(exc, error_type):
            return message, code
    return (
        "The background graph configuration is incompatible with this job.",
        "background_configuration_error",
    )


__all__ = ["BackgroundWorker"]
