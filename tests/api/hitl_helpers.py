import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from langgraph_openai_serve import GraphConfig, GraphRegistry


@dataclass
class GraphWithCounter:
    registry: GraphRegistry
    counter: dict[str, int]

    @property
    def count(self) -> int:
        return self.counter["count"]


def registry(
    model: str,
    graph: Any,
    streamable_node_names: list[str] | None = None,
) -> GraphRegistry:
    return GraphRegistry(
        registry={
            model: GraphConfig(
                graph=graph,
                streamable_node_names=streamable_node_names or [],
            )
        }
    )


def user_messages(content: str = "do it") -> list[dict[str, Any]]:
    return [{"role": "user", "content": content}]


def tool_calls(chunks: list[Any]) -> list[Any]:
    return [
        tool_call
        for chunk in chunks
        for tool_call in (chunk.choices[0].delta.tool_calls or [])
    ]


def stream_text(chunks: list[Any]) -> str:
    return "".join(chunk.choices[0].delta.content or "" for chunk in chunks)


def append_tool_response(
    messages: list[dict[str, Any]],
    tool_call: Any,
    response: Any,
    assistant_message: Any | None = None,
) -> None:
    assistant = (
        assistant_message.model_dump(exclude_none=True)
        if assistant_message is not None
        else _assistant_tool_call_message(tool_call)
    )
    messages.extend(
        [
            assistant,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(response),
            },
        ]
    )


def approval_registry(
    model: str,
    prompt: str,
    streamable_node_names: list[str] | None = None,
) -> GraphRegistry:
    def ask_for_approval(state):
        decision = interrupt({"prompt": prompt})
        outcome = "approved" if decision["approved"] else "rejected"
        return {"messages": [AIMessage(content=outcome)]}

    graph = (
        StateGraph(dict)
        .add_node("approval", ask_for_approval)
        .set_entry_point("approval")
        .add_edge("approval", END)
        .compile(checkpointer=InMemorySaver())
    )
    return registry(model, graph, streamable_node_names)


def approval_control_registry() -> GraphWithCounter:
    counter = {"count": 0}

    def review(state_):
        decision = interrupt({"prompt": "Execute?"})
        return Command(goto="execute" if decision["approved"] else "reject")

    def execute(state_):
        counter["count"] += 1
        request = state_["messages"][-1].content
        return {"messages": [AIMessage(content=f"Executed: {request}")]}

    def reject(state_):
        request = state_["messages"][-1].content
        return {
            "messages": [
                AIMessage(content=f"Cancelled without executing the request: {request}")
            ]
        }

    graph = (
        StateGraph(dict)
        .add_node("review", review)
        .add_node("execute", execute)
        .add_node("reject", reject)
        .set_entry_point("review")
        .add_edge("execute", END)
        .add_edge("reject", END)
        .compile(checkpointer=InMemorySaver())
    )
    return GraphWithCounter(registry=registry("hitl-demo", graph), counter=counter)


def _assistant_tool_call_message(tool_call: Any) -> dict[str, Any]:
    assert tool_call.id
    assert tool_call.function
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }
