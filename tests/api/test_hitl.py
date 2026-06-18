import json

from langgraph_openai_serve.hitl.openai import HITL_TOOL_NAME
from tests.api.hitl_helpers import (
    append_tool_response,
    approval_control_registry,
    approval_registry,
    stream_text,
    tool_calls,
    user_messages,
)


def test_official_openai_client_streams_and_resumes_interrupt(
    openai_client_factory,
) -> None:
    registry = approval_registry(
        "hitl-stream",
        "Approve streaming?",
        streamable_node_names=["approval"],
    )

    with openai_client_factory(registry) as client:
        messages = user_messages()
        interrupted_chunks = list(
            client.chat.completions.create(
                model="hitl-stream",
                messages=messages,
                stream=True,
            )
        )

        tool_deltas = tool_calls(interrupted_chunks)
        assert len(tool_deltas) == 1
        tool_call = tool_deltas[0]
        assert tool_call.index == 0
        assert tool_call.id
        assert tool_call.function
        assert tool_call.function.name == HITL_TOOL_NAME
        arguments = json.loads(tool_call.function.arguments or "")
        assert arguments["interrupts"][0]["value"] == {"prompt": "Approve streaming?"}
        assert interrupted_chunks[-1].choices[0].finish_reason == "tool_calls"

        append_tool_response(messages, tool_call, {"approved": True})
        resumed_chunks = list(
            client.chat.completions.create(
                model="hitl-stream",
                messages=messages,
                stream=True,
            )
        )

        assert stream_text(resumed_chunks) == "approved"
        assert resumed_chunks[-1].choices[0].finish_reason == "stop"


def test_hitl_approval_controls_execution(
    openai_client_factory,
) -> None:
    graph = approval_control_registry()

    with openai_client_factory(graph.registry) as client:

        def run_request(request: str, approved: bool) -> str | None:
            messages = user_messages(request)
            interrupted = client.chat.completions.create(
                model="hitl-demo",
                messages=messages,
            )
            choice = interrupted.choices[0]
            assert choice.finish_reason == "tool_calls"
            assistant = choice.message
            assert assistant.tool_calls
            tool_call = assistant.tool_calls[0]
            assert tool_call.function.name == HITL_TOOL_NAME
            arguments = json.loads(tool_call.function.arguments)
            assert arguments["interrupts"][0]["value"] == {"prompt": "Execute?"}
            append_tool_response(
                messages,
                tool_call,
                {"approved": approved},
                assistant_message=assistant,
            )
            resumed = client.chat.completions.create(
                model="hitl-demo",
                messages=messages,
            )
            return resumed.choices[0].message.content

        assert run_request("prepare a report", approved=False) == (
            "Cancelled without executing the request: prepare a report"
        )
        assert graph.count == 0

        assert run_request("prepare a report", approved=True) == (
            "Executed: prepare a report"
        )
        assert graph.count == 1
