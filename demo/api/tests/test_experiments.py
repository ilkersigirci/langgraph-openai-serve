from datetime import UTC, datetime
from types import FunctionType, SimpleNamespace
from typing import Any

import lgos_rag
import pytest
from _shared import run, score
from langfuse import Evaluation, Langfuse, RegressionError, RunnerContext
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import (
    EvaluatorFunction,
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


async def _double(*, item: LocalExperimentItem, **_kwargs: Any) -> int:
    if item["input"] == 1:
        msg = "the gateway is down"
        raise RuntimeError(msg)
    return item["input"] * 2


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


def test_each_case_reports_its_outcome_as_it_finishes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(ITEMS, max_failures=2)

    lines = capsys.readouterr().out.splitlines()
    # Cases finish concurrently, so only the count, not the order, is fixed.
    assert [line.partition("] ")[0] for line in lines] == ["[1/3", "[2/3", "[3/3"]
    assert sorted(line.partition("] ")[2] for line in lines) == [
        "✓ case 3",
        "✗ case 1 (task raised RuntimeError: the gateway is down)",
        "✗ case 2 (checks)",
    ]


def test_async_evaluators_score_cases() -> None:
    async def doubled(
        *, output: int, expected_output: Any, **_kwargs: Any
    ) -> Evaluation:
        return score("double", passed=output == expected_output)

    result = _run(ITEMS[2:], max_failures=0, evaluators=[doubled])

    assert [
        (check.name, check.value) for check in result.item_results[0].evaluations
    ] == [("double", 1.0), ("passed", 1.0)]


def verdict_named(**_kwargs: Any) -> Evaluation:
    return score("passed", passed=True)


def malformed(**_kwargs: Any) -> dict[str, Any]:
    return {"nme": "typo", "value": 0}


def unscored(**_kwargs: Any) -> None:
    return None


@pytest.mark.parametrize("evaluator", [verdict_named, malformed, unscored])
def test_an_evaluator_that_returns_no_valid_checks_fails_its_case(
    evaluator: FunctionType,
) -> None:
    # Another evaluator's passing check must not carry the case.
    with pytest.raises(RegressionError, match="1 of 1 cases failed") as error:
        _run(ITEMS[2:], max_failures=0, evaluators=[checks, evaluator])
    assert f"case 3: {evaluator.__name__}: " in str(error.value)


REPEATED: list[LocalExperimentItem] = [
    ITEMS[2],
    {**ITEMS[0], "metadata": ITEMS[2]["metadata"]},
]


@pytest.mark.parametrize(
    ("items", "data"),
    [
        pytest.param([], [], id="no-cases"),
        # The Langfuse action can bring an empty dataset.
        pytest.param(ITEMS, [], id="no-data"),
        pytest.param(REPEATED, REPEATED, id="repeated-id"),
    ],
)
def test_a_run_needs_cases_with_unique_ids(
    items: list[LocalExperimentItem], data: list[LocalExperimentItem]
) -> None:
    with pytest.raises(ValueError, match=r"no cases|repeats case IDs"):
        run(
            RunnerContext(client=Langfuse(tracing_enabled=False), data=data),
            name="unit",
            description="unit",
            items=items,
            task=_double,
            evaluators=[checks],
            max_failures=0,
            metadata={},
        )


class _Datasets:
    """A hosted dataset store; a rival run may sync its own cases mid-sync."""

    def __init__(self, *hosted: DatasetItem, rival: DatasetItem | None = None) -> None:
        self.items = {item.id: item for item in hosted}
        self.writes: list[tuple[str, DatasetStatus]] = []
        self.rival = rival

    def create_dataset(self, **_kwargs: Any) -> None:
        pass

    def get_dataset(self, name: str) -> SimpleNamespace:
        active = [i for i in self.items.values() if i.status == DatasetStatus.ACTIVE]
        return SimpleNamespace(items=active)

    def create_dataset_item(self, **fields: Any) -> DatasetItem:
        item = _hosted(**fields)
        self.items[item.id] = item
        self.writes.append((item.id, item.status))
        if self.rival:
            # The rival archives this run's cases and adds its own.
            for item_id, other in self.items.items():
                self.items[item_id] = other.model_copy(
                    update={"status": DatasetStatus.ARCHIVED}
                )
            self.items[self.rival.id], self.rival = self.rival, None
        return item


def _hosted(**fields: Any) -> DatasetItem:
    now = datetime.now(UTC)
    return DatasetItem.model_validate(
        {
            "status": DatasetStatus.ACTIVE,
            "dataset_id": "dataset",
            "dataset_name": "unit",
            "created_at": now,
            "updated_at": now,
            "media_references": [],
            **fields,
        }
    )


def _sync_and_run(
    datasets: _Datasets, monkeypatch: pytest.MonkeyPatch
) -> ExperimentResult:
    # A closed local port fails the runner's project lookup at once.
    client = Langfuse(
        public_key="unit",
        secret_key="unit",
        base_url="http://127.0.0.1:9",
        tracing_enabled=False,
    )
    for method in ("create_dataset", "get_dataset", "create_dataset_item"):
        monkeypatch.setattr(client, method, getattr(datasets, method))
    return run(
        RunnerContext(client=client),
        name="unit",
        description="unit",
        items=ITEMS[2:],
        task=_double,
        evaluators=[checks],
        max_failures=0,
        metadata={},
    )


def test_a_run_evaluates_the_cases_it_synced(monkeypatch: pytest.MonkeyPatch) -> None:
    rival = _hosted(
        id="unit-rival", input=4, expected_output=8, metadata={"case_id": "rival"}
    )
    result = _sync_and_run(_Datasets(rival=rival), monkeypatch)

    (item_result,) = result.item_results
    assert isinstance(item_result.item, DatasetItem)
    assert item_result.item.metadata == {"case_id": "case 3"}
    assert item_result.output == 6


def test_a_sync_writes_only_changed_and_removed_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = _hosted(
        id="unit-removed", input=5, expected_output=10, metadata={"case_id": "case 5"}
    )
    datasets = _Datasets(removed)
    _sync_and_run(datasets, monkeypatch)
    (added,) = [item_id for item_id, _ in datasets.writes if item_id != removed.id]
    writes = list(datasets.writes)

    # A second sync of the same cases rewrites nothing.
    _sync_and_run(datasets, monkeypatch)

    assert writes == [
        (added, DatasetStatus.ACTIVE),
        (removed.id, DatasetStatus.ARCHIVED),
    ]
    assert datasets.writes == writes


@pytest.mark.parametrize(
    ("question", "queries"),
    [
        # A vague message gets a clarifying question without a search.
        ("Why is it failing?", ["LGOS failure causes"]),
        # A scope hint follows a search of the selected scope.
        ("What does DEMO_GRAPH_GATEWAY_API_KEY set?", []),
    ],
)
def test_lgos_rag_scores_the_route_of_reply_shape_cases(
    question: str, queries: list[str]
) -> None:
    (item,) = [i for i in lgos_rag.ITEMS if i["metadata"]["case_id"] == question]

    scores = lgos_rag.checks(
        input=item["input"],
        expected_output=item["expected_output"],
        output={
            # The reply passes either reply check; only the route is wrong.
            "answer": "What fails? Set Docs scope to All to search every page.",
            "queries": queries,
            "sources": [],
            "source_links": [],
            "citations": [],
        },
    )

    assert {check.name: check.value for check in scores} == {
        "route": 0.0,
        "reply": 1.0,
    }
