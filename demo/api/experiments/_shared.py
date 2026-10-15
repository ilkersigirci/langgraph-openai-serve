"""
Langfuse plumbing shared by the demo's experiments.

An experiment script follows the Langfuse experiment runner convention: a task
and evaluators with the runner's keyword arguments, and an `experiment(context)`
entry point that returns `run(...)`. Its Git-tracked cases are local experiment
items, its `case_id` function names a case by its input, and its Pydantic models
define the shape of a case's input and expected output.

In Langfuse, `run` syncs the cases into the `<name>/regression` dataset and runs
the items it synced. It also keeps a `<name>/curated` dataset, with the same
case schemas, for cases that testers curate in the Langfuse UI. A context that
brings data, such as the Langfuse action's dataset, `main --dataset`, or `main`
without Langfuse, runs that data instead. `run` scores each case and the run,
and raises `RegressionError` when too many cases fail.

The Langfuse experiment action skips files whose names start with `_` and puts
the experiment's folder on `sys.path`, so experiments import this module as
`_shared`, both there and when run as scripts.
"""

import argparse
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
from pydantic import BaseModel

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


def _sync_datasets(
    langfuse: Langfuse,
    *,
    name: str,
    description: str,
    items: Sequence[LocalExperimentItem],
    case_id: Callable[[Any], str],
    input_schema: type[BaseModel],
    expected_output_schema: type[BaseModel],
) -> list[DatasetItem]:
    """Mirror `items` into the regression dataset, and return the hosted ones."""
    regression = f"{name}/regression"
    schemas = {
        "input_schema": input_schema.model_json_schema(),
        "expected_output_schema": expected_output_schema.model_json_schema(),
    }
    # Creating an existing dataset updates its description and schemas.
    langfuse.create_dataset(
        name=regression,
        description=(
            f"{description} Mirrors the Git-tracked cases of "
            "demo/api/experiments; every run there overwrites edits made here."
        ),
        **schemas,
    )
    langfuse.create_dataset(
        name=f"{name}/curated",
        description=(
            f"{description} Curated in Langfuse; run it with "
            f"`just demo/experiment {name} --dataset {name}/curated`."
        ),
        **schemas,
    )
    # Langfuse lists only active items; archived ones stay out of runs.
    hosted = {item.id: item for item in langfuse.get_dataset(regression).items}
    synced = {}
    for item in items:
        # Item IDs are unique across a project's datasets, so each hashes its
        # dataset's name too; the hash also keeps the name's slash out of URLs.
        key = f"{regression}\n{case_id(item['input'])}"
        item_id = hashlib.sha256(key.encode()).hexdigest()[:32]
        old = hosted.get(item_id)
        new = (item["input"], item.get("expected_output"), item.get("metadata"))
        # Rewriting an unchanged item would add a dataset version for nothing.
        if old is not None and new == (old.input, old.expected_output, old.metadata):
            synced[item_id] = old
        else:
            # The active status restores a case archived earlier.
            synced[item_id] = langfuse.create_dataset_item(
                dataset_name=regression,
                id=item_id,
                input=item["input"],
                expected_output=item.get("expected_output"),
                metadata=item.get("metadata"),
                status=DatasetStatus.ACTIVE,
            )
    for item_id, old in hosted.items():
        if item_id not in synced:
            langfuse.create_dataset_item(
                dataset_name=regression,
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

    def __init__(self, total: int, case_id: Callable[[Any], str]) -> None:
        self.total = total
        self.case_id = case_id
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
                    self.case_id(_local(item)["input"]),
                    f"task raised {type(error).__name__}: {error}",
                )
                raise
            return output

        return run_case

    def passed(
        self,
        *,
        input: Any = None,  # ruff: ignore[builtin-argument-shadowing] - Langfuse passes the item input by this name.
        evaluations: list[Evaluation],
        **_kwargs: Any,
    ) -> Evaluation:
        """Score a case as passed when it has checks and passed every one."""
        # The runner calls this composite evaluator last for each scored case.
        failed = [check.name for check in evaluations if check.value != 1]
        self._report(self.case_id(input), ", ".join(failed))
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


def run(  # ruff: ignore[too-many-arguments] - Keyword-only parts of one experiment definition.
    context: RunnerContext,
    *,
    name: str,
    description: str,
    items: Sequence[LocalExperimentItem],
    case_id: Callable[[Any], str],
    input_schema: type[BaseModel],
    expected_output_schema: type[BaseModel],
    task: Task,
    evaluators: list[EvaluatorFunction],
    max_failures: int,
    metadata: dict[str, str],
) -> ExperimentResult:
    """
    Run the cases as a Langfuse experiment and gate the result.

    `case_id` names a case by its input, and Langfuse validates every hosted
    case against the schemas. Each evaluator returns one `score` per check that
    applies, or fails the case under its own name when it raises or returns
    anything else, and the case is scored `passed` when it passes them all. The
    run is scored with `pass_rate`, and raises `RegressionError` when more than
    `max_failures` cases fail.
    """
    case_ids = [case_id(item["input"]) for item in items]
    if not case_ids:
        msg = f"Experiment {name} has no cases."
        raise ValueError(msg)
    if repeated := sorted({case for case in case_ids if case_ids.count(case) > 1}):
        msg = f"Experiment {name} repeats case IDs: {repeated}"
        raise ValueError(msg)
    data: ExperimentData = (
        _sync_datasets(
            context.client,
            name=name,
            description=description,
            items=items,
            case_id=case_id,
            input_schema=input_schema,
            expected_output_schema=expected_output_schema,
        )
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

    progress = _Progress(len(data), case_id)
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
        f"{case_id(_local(item)['input'])}: {failure}"
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="run this Langfuse dataset instead of the Git-tracked cases"
    )
    dataset = parser.parse_args().dataset
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        client = get_client()
        data = client.get_dataset(dataset).items if dataset else None
        context = RunnerContext(client=client, data=data)
    elif dataset:
        parser.error("--dataset needs the Langfuse keys")
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
