"""
Langfuse plumbing shared by the demo's experiments.

An experiment script follows the Langfuse experiment runner convention: a task
and evaluators with the runner's keyword arguments, and an `experiment(context)`
entry point that returns `run(...)`. Its cases are local dataset items, each
with a stable `case_id` in its metadata. `run` syncs them into a hosted dataset
named after the experiment, runs it, scores each case and the run, and raises
`RegressionError` when too many cases fail. When the context brings its own
data, as `main` does without Langfuse, `run` runs that data instead.

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
from typing import Any

from langfuse import Evaluation, Langfuse, RegressionError, RunnerContext, get_client
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import (
    EvaluatorFunction,
    ExperimentItem,
    ExperimentItemResult,
    ExperimentResult,
    LocalExperimentItem,
    TaskFunction,
)

# Bounds the graph runs at once, so a run stays within model rate limits.
MAX_CONCURRENCY = 8


def score(name: str, *, passed: bool, comment: str | None = None) -> Evaluation:
    """Return a pass-or-fail score for one check."""
    return Evaluation(
        name=name, value=float(passed), data_type="BOOLEAN", comment=comment
    )


def item_input(item: ExperimentItem) -> Any:
    """Return an item's input; hosted items are objects, local ones dictionaries."""
    return item.input if isinstance(item, DatasetItem) else item["input"]


def _case_id(item: ExperimentItem) -> str:
    metadata = item.metadata if isinstance(item, DatasetItem) else item.get("metadata")
    return (metadata or {})["case_id"]


def _sync_dataset(
    langfuse: Langfuse,
    name: str,
    description: str,
    items: Sequence[LocalExperimentItem],
) -> list[DatasetItem]:
    """Make the hosted dataset's items exactly `items`, and return them."""
    # Creating an existing dataset updates its description.
    langfuse.create_dataset(name=name, description=description)
    # Langfuse lists only active items; archived ones stay out of runs.
    hosted = {item.id: item for item in langfuse.get_dataset(name).items}
    # Item IDs are unique across a project's datasets, so they name the dataset.
    wanted = {
        f"{name}-{hashlib.sha256(_case_id(item).encode()).hexdigest()[:16]}": item
        for item in items
    }
    for item_id, item in wanted.items():
        old = hosted.get(item_id)
        new = (item["input"], item.get("expected_output"), item["metadata"])
        # Rewriting an unchanged item would add a dataset version for nothing.
        if old is None or new != (old.input, old.expected_output, old.metadata):
            # The active status restores a case archived earlier.
            langfuse.create_dataset_item(
                dataset_name=name,
                id=item_id,
                input=item["input"],
                expected_output=item.get("expected_output"),
                metadata=item["metadata"],
                status=DatasetStatus.ACTIVE,
            )
    for item_id, old in hosted.items():
        if item_id not in wanted:
            langfuse.create_dataset_item(
                dataset_name=name,
                id=item_id,
                input=old.input,
                expected_output=old.expected_output,
                metadata=old.metadata,
                status=DatasetStatus.ARCHIVED,
            )
    return langfuse.get_dataset(name).items


def _failing_on_error(evaluator: EvaluatorFunction) -> EvaluatorFunction:
    """Return the evaluator, scoring an error as a failed check named after it."""
    name = getattr(evaluator, "__name__", type(evaluator).__name__)

    # The runner would only log the error and leave the case unscored, which
    # hides it from a filter on failed cases.
    @functools.wraps(evaluator)
    async def evaluate(**kwargs: Any) -> Any:
        try:
            result = evaluator(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:  # ruff: ignore[blind-except] - Any evaluator error fails its case.
            return score(name, passed=False, comment=f"{type(error).__name__}: {error}")
        return result

    return evaluate


def _all_checks_passed(*, evaluations: list[Evaluation], **_kwargs: Any) -> Evaluation:
    """Score a case as passed when it has checks and passed every one."""
    failed = [check.name for check in evaluations if check.value != 1]
    return score(
        "passed",
        passed=bool(evaluations) and not failed,
        comment=f"failed: {failed}" if failed else None,
    )


def _passed(result: ExperimentItemResult) -> bool:
    return any(
        check.name == "passed" and check.value == 1 for check in result.evaluations
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
        if check.name != "passed" and check.value != 1
    ]
    return "; ".join(failed) or "unscored; see its trace or the log"


def run(
    context: RunnerContext,
    *,
    name: str,
    description: str,
    items: Sequence[LocalExperimentItem],
    task: TaskFunction,
    evaluators: list[EvaluatorFunction],
    max_failures: int,
    metadata: dict[str, str],
) -> ExperimentResult:
    """
    Run the cases as a Langfuse experiment and gate the result.

    Each evaluator returns one `score` per check, or fails under its own name
    when it raises, and the case is scored `passed` when it passes them all.
    The run is scored with `pass_rate`, and raises `RegressionError` when more
    than `max_failures` cases fail.
    """
    case_ids = [_case_id(item) for item in items]
    if not case_ids:
        msg = f"Experiment {name} has no cases."
        raise ValueError(msg)
    if repeated := sorted({case for case in case_ids if case_ids.count(case) > 1}):
        msg = f"Experiment {name} repeats case IDs: {repeated}"
        raise ValueError(msg)
    data = context.data
    if data is None:
        data = _sync_dataset(context.client, name, description, items)

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

    result = context.run_experiment(
        name=name,
        description=description,
        data=data,
        task=task,
        evaluators=[_failing_on_error(evaluator) for evaluator in evaluators],
        composite_evaluator=_all_checks_passed,
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
