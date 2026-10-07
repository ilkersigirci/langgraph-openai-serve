"""Opt-in checks through the running gateway; these incur model usage."""

import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] - Test-only subprocess with explicit arguments and no shell.
from uuid import uuid4

import pytest
from openai import AsyncOpenAI

from tests.support import answer_deltas

pytestmark = pytest.mark.integration


def workspace_command(*args: str) -> str:
    return subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        ["docker", "exec", "lgos-api-coding-agent", *args],  # ruff: ignore[start-process-with-partial-path] - The integration test uses Docker from the developer PATH.
        text=True,
        timeout=10,
    )


async def test_shell_execution_edits_streaming_and_persistent_follow_up() -> None:
    root = os.environ["DEMO_GATEWAY_HOST_URL"].rstrip("/")
    prefix = "/openai/v1" if os.environ["OPENAI_GATEWAY_TYPE"] == "bifrost" else "/v1"
    directory = f"/workspace/lgos-smoke-{uuid4().hex}"
    path = f"{directory}/token.txt"
    prompt = (
        f"Create {directory}. Run Python to generate secrets.token_hex(16) and "
        f"write it followed by a newline to {path}. Read the file back using "
        "a shell command. In your final answer print only the generated token."
    )
    try:
        async with AsyncOpenAI(
            base_url=root + prefix,
            api_key=os.environ["OPENAI_GATEWAY_API_KEY"],
            timeout=660,
            max_retries=0,
        ) as client:
            stream = await client.responses.create(
                model="lgos-api-coding-agent/coding-agent",
                input=prompt,
                stream=True,
                store=False,
            )
            events = [event async for event in stream]
            assert events[-1].type == "response.completed", events[-1].type
            response = events[-1].response
            answer = "".join(answer_deltas(events))
            token = workspace_command("cat", path).strip()
            assert re.fullmatch(r"[0-9a-f]{32}", token)
            assert token in answer
            final = [
                item
                for item in response.output
                if item.type == "message" and item.phase == "final_answer"
            ]
            assert answer == "".join(
                part.text
                for item in final
                for part in item.content
                if part.type == "output_text"
            )
            assert any(
                item.type == "message" and item.phase == "commentary"
                for item in response.output
            )
            assert response.usage
            assert response.usage.total_tokens > 0
            follow_up = await client.responses.create(
                model="lgos-api-coding-agent/coding-agent",
                store=False,
                input=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": "Append a second line containing follow-up to the same file. Read it back and report both lines.",
                    },
                ],
            )
            assert follow_up.status == "completed"
            assert token in follow_up.output_text
            assert workspace_command("cat", path) == f"{token}\nfollow-up\n"
    finally:
        workspace_command("rm", "-rf", directory)
