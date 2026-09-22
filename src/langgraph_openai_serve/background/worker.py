"""Checkpoint-aware graph execution for native background workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anyio import CancelScope, get_cancelled_exc_class
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
from langgraph_openai_serve.background.contracts import (
    BackgroundSettings,
    RetryableJobError,
)
from langgraph_openai_serve.background.responses import (
    failed_response,
    is_stored_response,
    output_response,
    response_json,
)
from langgraph_openai_serve.background.store import (
    ResponseStatus,
    ResponseStore,
    StoredRun,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphNotFoundError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.interrupt.coordination import RunBusyError
from langgraph_openai_serve.graph.runner import (
    BackgroundCheckpointIncompleteError,
    BackgroundGraphInterruptedError,
    run_background_graph,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from langgraph_openai_serve.graph.request import GraphRequest


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    request: ResponseCreateRequest
    graph_config: GraphConfig
    graph_request: GraphRequest
    messages: list[BaseMessage]


_PERMANENT_ERRORS = (
    BackgroundCheckpointIncompleteError,
    BackgroundGraphInterruptedError,
    GraphConfigurationError,
    GraphNotFoundError,
    UnsupportedResponsesOutputError,
    UnsupportedResponsesRequestError,
    ValidationError,
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
        if not isinstance(graphs, GraphRegistry):
            msg = "graphs must be a GraphRegistry."
            raise TypeError(msg)
        if not isinstance(store, ResponseStore):
            msg = "store must implement ResponseStore."
            raise TypeError(msg)
        self.graphs = graphs
        self.store = store
        self.settings = settings or BackgroundSettings()

    async def execute(self, response_id: str) -> None:
        """Execute one engine delivery, resuming from LangGraph checkpoints."""
        await self._with_retry(response_id, finalize_only=False)

    async def finalize(self, response_id: str) -> None:
        """Publish checkpointed output or failure after the engine exhausts retries."""
        await self._with_retry(response_id, finalize_only=True)

    async def _with_retry(self, response_id: str, *, finalize_only: bool) -> None:
        try:
            await self._deliver(response_id, finalize_only=finalize_only)
        except (RetryableJobError, get_cancelled_exc_class()):
            raise
        except Exception as exc:
            operation = "failure finalization" if finalize_only else "graph execution"
            msg = f"Background {operation} needs another engine attempt."
            raise RetryableJobError(msg) from exc

    async def _deliver(self, response_id: str, *, finalize_only: bool) -> None:
        run = await self.store.get_internal(response_id)
        if run is None:
            return
        if run.terminal:
            await self._cleanup_run(run)
            return

        prepared = await self._prepare_or_fail(run)
        if prepared is None:
            return
        coordinator = prepared.graph_config.run_coordinator
        if coordinator is None:
            await self._publish_failure(
                run,
                message="The background model has no cross-worker coordinator.",
                code="background_configuration_error",
            )
            return

        try:
            async with coordinator(run.checkpoint_thread_id):
                await self._run_coordinated(
                    run,
                    prepared,
                    finalize_only=finalize_only,
                )
        except RunBusyError as exc:
            raise RetryableJobError(str(exc)) from exc
        except get_cancelled_exc_class():
            with CancelScope(shield=True):
                latest = await self.store.get_internal(run.response_id)
            if latest is not None and latest.status is ResponseStatus.CANCELLED:
                return
            raise

    async def _run_coordinated(
        self,
        run: StoredRun,
        prepared: _PreparedRun,
        *,
        finalize_only: bool,
    ) -> None:
        current = await self.store.get_internal(run.response_id)
        if current is None:
            return
        if current.terminal:
            await self._cleanup_locked(current, prepared.graph_config)
            return
        if not finalize_only:
            current = await self.store.mark_in_progress(
                run.response_id,
                now=datetime.now(UTC),
            )
            if current is None:
                return
        try:
            terminal = await self._run_and_publish(
                current,
                prepared,
                finalize_only=finalize_only,
            )
        except _PERMANENT_ERRORS as exc:
            message, code = _public_failure(exc)
            terminal = await self._publish_failure(
                current,
                message=message,
                code=code,
            )
        if terminal is not None:
            await self._cleanup_locked(terminal, prepared.graph_config)

    async def _prepare_or_fail(
        self,
        run: StoredRun,
    ) -> _PreparedRun | None:
        try:
            prepared = self._prepare_run(run)
        except _PERMANENT_ERRORS as exc:
            message, code = _public_failure(exc)
            terminal = await self._publish_failure(run, message=message, code=code)
            if terminal is not None:
                await self._cleanup_run(terminal)
            return None
        policy = prepared.graph_config.background
        if policy is None or policy.version != run.graph_version:
            terminal = await self._publish_failure(
                run,
                message=(
                    "The registered background graph version is incompatible "
                    "with this job."
                ),
                code="background_graph_incompatible",
            )
            if terminal is not None:
                await self._cleanup_run(terminal)
            return None
        return prepared

    def _prepare_run(self, run: StoredRun) -> _PreparedRun:
        request = ResponseCreateRequest.model_validate(run.envelope)
        graph_config = self.graphs.get_graph(run.model)
        validate_tools(request, graph_config.server_tools)
        graph_request, messages, _ = decode_responses_request(
            request,
            graph_config.server_tools,
        )
        return _PreparedRun(request, graph_config, graph_request, messages)

    async def _run_and_publish(
        self,
        run: StoredRun,
        prepared: _PreparedRun,
        *,
        finalize_only: bool = False,
    ) -> StoredRun | None:
        result = await run_background_graph(
            prepared.graph_request,
            prepared.messages,
            prepared.graph_config,
            checkpoint_thread_id=run.checkpoint_thread_id,
            finalize_only=finalize_only,
            initial_message_count=run.initial_message_count,
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
        response = output_response(
            prepared.request,
            result.message,
            response_id=run.response_id,
            created_at=int(run.created_at.timestamp()),
            server_tools=server_tools,
            root_messages=result.root_messages,
            initial_call_ids=frozenset(run.initial_call_ids),
        )
        published = await self.store.publish_terminal(
            run.response_id,
            response_json(response),
            now=datetime.now(UTC),
            result_retention=self.settings.result_retention_for(
                stored=is_stored_response(response)
            ),
            idempotency_retention=self.settings.idempotency_retention,
        )
        if published is not None:
            return published
        latest = await self.store.get_internal(run.response_id)
        if latest is not None and latest.terminal:
            return latest
        msg = "Background terminal publication lost its persisted run."
        raise RetryableJobError(msg)

    async def _publish_failure(
        self,
        run: StoredRun,
        *,
        message: str,
        code: str,
    ) -> StoredRun | None:
        if run.response is None:
            msg = "Background failure publication lost its Response snapshot."
            raise RetryableJobError(msg)
        response = failed_response(
            Response.model_validate(run.response), message=message
        )
        published = await self.store.publish_terminal(
            run.response_id,
            response_json(response),
            now=datetime.now(UTC),
            result_retention=self.settings.result_retention_for(
                stored=is_stored_response(response)
            ),
            idempotency_retention=self.settings.idempotency_retention,
        )
        if published is not None:
            logger.warning(
                "background.response_failed",
                extra={"response_id": run.response_id, "failure_code": code},
            )
            return published
        latest = await self.store.get_internal(run.response_id)
        if latest is not None and latest.terminal:
            return latest
        msg = "Background failure publication lost its persisted run."
        raise RetryableJobError(msg)

    async def cleanup_once(self, *, limit: int | None = None) -> int:
        """Delete terminal checkpoint lineages in one bounded pass."""
        runs = await self.store.claim_cleanup_ready(
            now=datetime.now(UTC),
            limit=limit or self.settings.maintenance_batch_size,
        )
        cleaned = 0
        for run in runs:
            try:
                cleaned += int(await self._cleanup_run(run))
            except RunBusyError:
                continue
            except Exception:
                logger.exception(
                    "background.checkpoint_cleanup_failed",
                    extra={"response_id": run.response_id},
                )
                continue
        return cleaned

    async def maintain(self) -> dict[str, int]:
        """Run bounded cleanup on the engine's recurring schedule."""
        cleaned = await self.cleanup_once()
        expired = await self.store.expire(
            now=datetime.now(UTC),
            limit=self.settings.maintenance_batch_size,
        )
        return {"cleaned": cleaned, "expired": expired}

    async def _cleanup_run(self, run: StoredRun) -> bool:
        if not run.cleanup_pending:
            return False
        try:
            graph_config = self.graphs.get_graph(run.model)
        except GraphNotFoundError:
            return await self._abandon_cleanup(run)
        coordinator = graph_config.run_coordinator
        if coordinator is None:
            return await self._abandon_cleanup(run)
        async with coordinator(run.checkpoint_thread_id):
            current = await self.store.get_internal(run.response_id)
            if current is None or not current.terminal or not current.cleanup_pending:
                return False
            return await self._cleanup_locked(current, graph_config)

    async def _cleanup_locked(
        self,
        run: StoredRun,
        graph_config: GraphConfig,
    ) -> bool:
        try:
            graph = await graph_config.resolve_graph()
        except GraphConfigurationError:
            return await self._abandon_cleanup(run)
        checkpointer = graph.checkpointer
        if not isinstance(checkpointer, BaseCheckpointSaver):
            return await self._abandon_cleanup(run)
        await checkpointer.adelete_thread(run.checkpoint_thread_id)
        return await self.store.finish_cleanup(
            run.response_id,
            now=datetime.now(UTC),
        )

    async def _abandon_cleanup(self, run: StoredRun) -> bool:
        abandoned = await self.store.abandon_cleanup(
            run.response_id,
            now=datetime.now(UTC),
        )
        if abandoned:
            logger.warning(
                "background.checkpoint_cleanup_abandoned",
                extra={"response_id": run.response_id, "model": run.model},
            )
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
