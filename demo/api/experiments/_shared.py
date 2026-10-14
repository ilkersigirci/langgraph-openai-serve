"""
Langfuse plumbing shared by the demo's experiments.

An experiment script follows the Langfuse experiment runner convention: a task
and evaluators with the runner's keyword arguments, and an `experiment(context)`
entry point that returns `run(...)`. Its cases are local experiment items, each
with a stable `case_id` in its metadata. In Langfuse, `run` syncs them into a
hosted dataset named after the experiment and runs the items it synced; a
context that brings data, such as the Langfuse action's dataset or `main`
without Langfuse, runs that data instead. `run` scores each case and the run,
and raises `RegressionError` when too many cases fail.

The Langfuse experiment action skips files whose names start with `_` and puts
the experiment's folder on `sys.path`, so experiments import this module as
`_shared`, both there and when run as scripts.
"""

import functools
import hashlib
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langfuse import Evaluation, Langfuse, RegressionError, RunnerContext, get_client
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import (
    EvaluatorFunction,
    ExperimentData,
    ExperimentItem,
    ExperimentItemResult,
    ExperimentResult,
    LocalExperimentItem,
    TaskFunction,
)

# Bounds the graph runs at once, so a run stays within model rate limits.
MAX_CONCURRENCY = 8
# The composite score that holds each case's verdict; no check may use it.
VERDICT = "passed"


def score(name: str, *, passed: bool, comment: str | None = None) -> Evaluation:
    """Return a pass-or-fail score for one check."""
    return Evaluation(
        name=name, value=float(passed), data_type="BOOLEAN", comment=comment
    )


class Task(Protocol):
    """An experiment task: it runs one case and returns what the checks need."""

    def __call__(self, *, item: LocalExperimentItem, **kwargs: Any) -> Any: ...


def _local(item: ExperimentItem) -> LocalExperimentItem:
    """Return an item as a local one; hosted dataset items are objects."""
    if isinstance(item, DatasetItem):
        return {
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": item.metadata,
        }
    return item


def _case_id(item: ExperimentItem) -> str:
    return (_local(item).get("metadata") or {})["case_id"]


def _sync_dataset(
    langfuse: Langfuse,
    name: str,
    description: str,
    items: Sequence[LocalExperimentItem],
) -> list[DatasetItem]:
    """Make the hosted dataset's items exactly `items`, and return the hosted ones."""
    # Creating an existing dataset updates its description.
    langfuse.create_dataset(name=name, description=description)
    # Langfuse lists only active items; archived ones stay out of runs.
    hosted = {item.id: item for item in langfuse.get_dataset(name).items}
    synced = {}
    for item in items:
        # Item IDs are unique across a project's datasets, so each names it.
        case_id = _case_id(item)
        item_id = f"{name}-{hashlib.sha256(case_id.encode()).hexdigest()[:16]}"
        old = hosted.get(item_id)
        new = (item["input"], item.get("expected_output"), item.get("metadata"))
        # Rewriting an unchanged item would add a dataset version for nothing.
        if old is not None and new == (old.input, old.expected_output, old.metadata):
            synced[item_id] = old
        else:
            # The active status restores a case archived earlier.
            synced[item_id] = langfuse.create_dataset_item(
                dataset_name=name,
                id=item_id,
                input=item["input"],
                expected_output=item.get("expected_output"),
                metadata=item.get("metadata"),
                status=DatasetStatus.ACTIVE,
            )
    for item_id, old in hosted.items():
        if item_id not in synced:
            langfuse.create_dataset_item(
                dataset_name=name,
                id=item_id,
                input=old.input,
                expected_output=old.expected_output,
                metadata=old.metadata,
                status=DatasetStatus.ARCHIVED,
            )
    # Rereading the dataset could return the cases a concurrent run synced.
    return list(synced.values())


async def _checks(evaluator: EvaluatorFunction, **kwargs: Any) -> list[Evaluation]:
    """Return an evaluator's checks, raising when it returns anything else."""
    result = evaluator(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    # Langfuse accepts one check or a list, as objects or dictionaries.
    checks = [
        Evaluation(**check) if isinstance(check, dict) else check
        for check in (result if isinstance(result, list) else [result])
    ]
    for check in checks:
        if not isinstance(check, Evaluation):
            msg = f"returned {check!r}, not a check"
            raise TypeError(msg)
        if check.name == VERDICT:
            msg = f"named a check {VERDICT!r}, the case verdict"
            raise ValueError(msg)
    return checks


def _validated(evaluator: EvaluatorFunction) -> EvaluatorFunction:
    """Wrap the evaluator so that anything but checks fails the case under its name."""
    name = getattr(evaluator, "__name__", type(evaluator).__name__)

    # The runner would only log an error or drop a malformed result, leaving
    # the other evaluators to decide the case.
    @functools.wraps(evaluator)
    async def evaluate(**kwargs: Any) -> list[Evaluation]:
        try:
            return await _checks(evaluator, **kwargs)
        except Exception as error:  # ruff: ignore[blind-except] - Any evaluator error fails its case.
            return [
                score(name, passed=False, comment=f"{type(error).__name__}: {error}")
            ]

    return evaluate


class _Progress:
    """Print a line per finished case; the runner reports only at the end."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0

    def _report(self, case_id: str, failure: str | None) -> None:
        self.done += 1
        outcome = f"✗ {case_id} ({failure})" if failure else f"✓ {case_id}"
        print(f"[{self.done}/{self.total}] {outcome}", flush=True)

    def task(self, task: Task) -> TaskFunction:
        """Return the task, reporting its case when it raises."""

        async def run_case(*, item: ExperimentItem, **kwargs: Any) -> Any:
            try:
                # Scripts read every item the same way, hosted or local.
                output = task(item=_local(item), **kwargs)
                if inspect.isawaitable(output):
                    output = await output
            except Exception as error:
                self._report(
                    _case_id(item), f"task raised {type(error).__name__}: {error}"
                )
                raise
            return output

        return run_case

    def passed(
        self,
        *,
        evaluations: list[Evaluation],
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Evaluation:
        """Score a case as passed when it has checks and passed every one."""
        # The runner calls this composite evaluator last for each scored case.
        failed = [check.name for check in evaluations if check.value != 1]
        self._report((metadata or {})["case_id"], ", ".join(failed))
        return score(
            VERDICT,
            passed=bool(evaluations) and not failed,
            comment=f"failed: {failed}" if failed else None,
        )


def _passed(result: ExperimentItemResult) -> bool:
    return any(
        check.name == VERDICT and check.value == 1 for check in result.evaluations
    )


def _failure(result: ExperimentItemResult | None) -> str | None:
    """Describe why a case failed, or return None when it passed."""
    # The runner drops a case whose task raised; it logs and traces the error.
    if result is None:
        return "the task raised; see its trace or the log"
    if _passed(result):
        return None
    failed = [
        f"{check.name}: {check.comment}"
        for check in result.evaluations
        if check.name != VERDICT and check.value != 1
    ]
    return "; ".join(failed) or "unscored; see its trace or the log"


def run(
    context: RunnerContext,
    *,
    name: str,
    description: str,
    items: Sequence[LocalExperimentItem],
    task: Task,
    evaluators: list[EvaluatorFunction],
    max_failures: int,
    metadata: dict[str, str],
) -> ExperimentResult:
    """
    Run the cases as a Langfuse experiment and gate the result.

    Each evaluator returns one `score` per check that applies, or fails the case
    under its own name when it raises or returns anything else, and the case is
    scored `passed` when it passes them all. The run is scored with
    `pass_rate`, and raises `RegressionError` when more than `max_failures`
    cases fail.
    """
    case_ids = [_case_id(item) for item in items]
    if not case_ids:
        msg = f"Experiment {name} has no cases."
        raise ValueError(msg)
    if repeated := sorted({case for case in case_ids if case_ids.count(case) > 1}):
        msg = f"Experiment {name} repeats case IDs: {repeated}"
        raise ValueError(msg)
    data: ExperimentData = (
        _sync_dataset(context.client, name, description, items)
        if context.data is None
        else context.data
    )
    # The Langfuse action can bring an empty dataset.
    if not data:
        msg = f"Experiment {name} has no cases to run."
        raise ValueError(msg)

    def pass_rate(
        *, item_results: list[ExperimentItemResult], **_kwargs: Any
    ) -> Evaluation:
        # Results omit cases whose task raised, so divide by every case sent.
        passed = sum(map(_passed, item_results))
        return Evaluation(
            name="pass_rate",
            value=passed / len(data),
            comment=f"{passed} of {len(data)} cases passed",
        )

    progress = _Progress(len(data))
    result = context.run_experiment(
        name=name,
        description=description,
        data=data,
        task=progress.task(task),
        evaluators=[_validated(evaluator) for evaluator in evaluators],
        composite_evaluator=progress.passed,
        run_evaluators=[pass_rate],
        max_concurrency=MAX_CONCURRENCY,
        metadata=metadata,
    )
    results = {id(item_result.item): item_result for item_result in result.item_results}
    failures = [
        f"{_case_id(item)}: {failure}"
        for item in data
        if (failure := _failure(results.get(id(item))))
    ]
    if len(failures) > max_failures:
        raise RegressionError(
            result=result,
            metric="failed_cases",
            value=len(failures),
            threshold=max_failures,
            message="\n- ".join(
                [f"{len(failures)} of {len(data)} cases failed:", *failures]
            ),
        )
    return result


def main(
    experiment: Callable[[RunnerContext], ExperimentResult],
    items: Sequence[LocalExperimentItem],
) -> None:
    """Run an experiment script in Langfuse when configured, otherwise locally."""
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        context = RunnerContext(client=get_client())
    else:
        print("Langfuse is not configured; running the cases without recording.")
        # Placeholder keys register the disabled client, so each
        # `CallbackHandler` reuses it instead of warning about missing keys.
        client = Langfuse(
            public_key="local",
            secret_key="local",  # ruff: ignore[hardcoded-password-func-arg] - A placeholder; tracing is off.
            tracing_enabled=False,
        )
        context = RunnerContext(client=client, data=list(items))
    try:
        result = experiment(context)
    except RegressionError as error:
        # Show the run before its failing cases, then exit non-zero.
        print(error.result.format())
        sys.exit(str(error))
    finally:
        context.client.flush()
    print(result.format())
