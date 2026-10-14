from typing import Any

import pytest
from _shared import item_input, run, score
from langfuse import Evaluation, Langfuse, RegressionError, RunnerContext
from langfuse.experiment import (
    EvaluatorFunction,
    ExperimentItem,
    ExperimentResult,
    LocalExperimentItem,
)

ITEMS: list[LocalExperimentItem] = [
    {
        "input": value,
        "expected_output": value * 2,
        "metadata": {"case_id": f"case {value}"},
    }
    for value in (1, 2, 3)
]


async def _double(*, item: ExperimentItem, **_kwargs: Any) -> int:
    if item_input(item) == 1:
        msg = "the gateway is down"
        raise RuntimeError(msg)
    return item_input(item) * 2


def checks(
    *,
    input: Any,  # ruff: ignore[builtin-argument-shadowing] - Langfuse passes the item input by this name.
    output: int,
    expected_output: Any,
    **_kwargs: Any,
) -> Evaluation:
    if input == 2:
        msg = "a check broke"
        raise KeyError(msg)
    return score("double", passed=output == expected_output)


def _run(
    items: list[LocalExperimentItem],
    *,
    max_failures: int,
    evaluators: list[EvaluatorFunction] | None = None,
) -> ExperimentResult:
    return run(
        RunnerContext(client=Langfuse(tracing_enabled=False), data=items),
        name="unit",
        description="unit",
        items=items,
        task=_double,
        evaluators=evaluators or [checks],
        max_failures=max_failures,
        metadata={},
    )


def test_cases_whose_task_or_checks_raise_fail() -> None:
    result = _run(ITEMS, max_failures=2)

    # The runner drops the case whose task raised; the pass rate still counts it.
    assert [(run.name, run.value) for run in result.run_evaluations] == [
        ("pass_rate", pytest.approx(1 / 3))
    ]
    # The broken check fails its case under the evaluator's name.
    assert sorted(
        (check.name, check.value)
        for item in result.item_results
        for check in item.evaluations
    ) == [("checks", 0.0), ("double", 1.0), ("passed", 0.0), ("passed", 1.0)]
    with pytest.raises(RegressionError, match="2 of 3 cases failed") as error:
        _run(ITEMS, max_failures=1)
    assert "case 1: the task raised" in str(error.value)
    assert "case 2: checks: KeyError: 'a check broke'" in str(error.value)


def test_async_evaluators_score_cases() -> None:
    async def doubled(
        *, output: int, expected_output: Any, **_kwargs: Any
    ) -> Evaluation:
        return score("double", passed=output == expected_output)

    result = _run(ITEMS[2:], max_failures=0, evaluators=[doubled])

    assert [
        (check.name, check.value) for check in result.item_results[0].evaluations
    ] == [("double", 1.0), ("passed", 1.0)]


@pytest.mark.parametrize(
    "items",
    [
        pytest.param([], id="no-cases"),
        pytest.param(
            [ITEMS[2], {**ITEMS[0], "metadata": ITEMS[2]["metadata"]}], id="repeated-id"
        ),
    ],
)
def test_cases_must_exist_with_unique_ids(items: list[LocalExperimentItem]) -> None:
    with pytest.raises(ValueError, match=r"no cases|repeats case IDs"):
        _run(items, max_failures=0)
