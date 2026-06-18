"""Minimal tool approval workflow for the Chainlit HITL demo."""

import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class HitlState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    approved_tool: dict[str, Any]


def calculator(operation: str, a: float, b: float) -> str:
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif b == 0:
        return "Cannot divide by zero."
    else:
        result = a / b
    return f"{result:g}"


def weather(city: str) -> str:
    forecasts = {
        "istanbul": "Istanbul: 22C, partly cloudy.",
        "london": "London: 16C, cloudy.",
        "paris": "Paris: 20C, mostly sunny.",
        "tokyo": "Tokyo: 26C, humid.",
    }
    return forecasts.get(city.lower(), f"{city}: 21C, clear.")


def agent(state: HitlState) -> dict:
    """Create one demo tool call from the user message."""
    tool = _pick_tool(str(state["messages"][-1].content))
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "demo_tool_call",
                        "name": tool["name"],
                        "args": tool["args"],
                    }
                ],
            )
        ]
    }


def human_approval(state: HitlState) -> Command[Literal["run_tool", "__end__"]]:
    """Pause for human approval before executing the requested tool."""
    tool_call = state["messages"][-1].tool_calls[0]
    response = interrupt(
        {
            "kind": "tool_approval",
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"],
            },
            "config": {
                "allow_accept": True,
                "allow_edit": True,
                "allow_respond": True,
                "allow_ignore": True,
            },
        }
    )
    human_response = response[0] if isinstance(response, list) else response

    if human_response["type"] == "accept":
        return Command(goto="run_tool", update={"approved_tool": tool_call})

    if human_response["type"] == "edit":
        return Command(
            goto="run_tool",
            update={
                "approved_tool": {
                    "name": tool_call["name"],
                    "args": human_response["args"],
                }
            },
        )

    if human_response["type"] == "response":
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=str(human_response.get("args", "")))]
            },
        )

    return Command(
        goto=END,
        update={"messages": [AIMessage(content="Ignored the tool call.")]},
    )


def run_tool(state: HitlState) -> dict:
    tool = state["approved_tool"]
    if tool["name"] == "weather":
        result = weather(**tool["args"])
    else:
        result = calculator(**tool["args"])
    return {"messages": [AIMessage(content=f"Tool result: {result}")]}


def _pick_tool(text: str) -> dict[str, Any]:
    if "weather" in text.lower():
        city = next(
            (
                city
                for city in ["Istanbul", "London", "Paris", "Tokyo"]
                if city.lower() in text.lower()
            ),
            "Paris",
        )
        return {"name": "weather", "args": {"city": city}}

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/xX])\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return {
            "name": "calculator",
            "args": {"operation": "multiply", "a": 18, "b": 42},
        }

    operations = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "x": "multiply",
        "X": "multiply",
        "/": "divide",
    }
    return {
        "name": "calculator",
        "args": {
            "operation": operations[match.group(2)],
            "a": float(match.group(1)),
            "b": float(match.group(3)),
        },
    }


builder = StateGraph(HitlState)
builder.add_node("agent", agent)
builder.add_node("human_approval", human_approval)
builder.add_node("run_tool", run_tool)
builder.set_entry_point("agent")
builder.add_edge("agent", "human_approval")
builder.add_edge("run_tool", END)

hitl_graph = builder.compile(checkpointer=InMemorySaver())
