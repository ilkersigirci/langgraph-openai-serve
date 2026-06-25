from demo.api.graphs.interruptible import interruptible_graph
from langgraph.types import Command


def test_interrupt_showcase_approval_resume() -> None:
    config = {"configurable": {"thread_id": "showcase-approve"}}

    first = interruptible_graph.invoke(
        {"request": "Refund order ORDER-123"},
        config=config,
    )
    interrupt = first["__interrupt__"][0]

    assert interrupt.value == {
        "question": "Approve this agent action?",
        "request": "Refund order ORDER-123",
        "choices": ["approve", "reject"],
    }

    resumed = interruptible_graph.invoke(Command(resume="approve"), config=config)

    assert resumed["response"] == "Approved agent action: Refund order ORDER-123"


def test_interrupt_showcase_reject_resume() -> None:
    config = {"configurable": {"thread_id": "showcase-reject"}}

    interruptible_graph.invoke(
        {"request": "Refund order ORDER-456"},
        config=config,
    )
    resumed = interruptible_graph.invoke(Command(resume="reject"), config=config)

    assert resumed["response"] == "Rejected agent action: Refund order ORDER-456"
