import json
import os
import pty
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] - Test-only subprocess with explicit arguments and no shell.
import sys
import textwrap

import pytest


def test_json_logs_carry_request_context_from_any_logger() -> None:
    # dictConfig replaces process-wide logging, so configure a fresh interpreter.
    script = textwrap.dedent(
        """
        import logging

        from langgraph_openai_serve.core.logging import begin_log_context
        from langgraph_openai_serve.server.logging import configure_logging

        configure_logging("my_app", root_level=logging.WARNING)
        begin_log_context("request-123")
        logging.getLogger("my_app.graphs").info("node.finished", extra={"node": "generate"})
        logging.getLogger("noisy_dependency").info("dependency.info")
        logging.getLogger("uvicorn.error").warning(
            "server.event", extra={"color_message": "stale ANSI message"}
        )
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logging.getLogger("langgraph_openai_serve.test").exception("run.failed")
        """
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        [sys.executable, "-c", script], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0, result.stderr
    node, server, failure = map(json.loads, result.stdout.splitlines())
    assert node["logger"] == "my_app.graphs"
    assert node["level"] == "info"
    assert node["message"] == "node.finished"
    assert node["request_id"] == "request-123"
    assert node["node"] == "generate"
    assert node["timestamp"]
    assert server["logger"] == "uvicorn.error"
    assert "color_message" not in server
    assert failure["level"] == "error"
    assert "RuntimeError: boom" in failure["exception"]


# The NO_COLOR convention (https://no-color.org) turns off terminal colors.
@pytest.mark.parametrize(
    ("variables", "colored"),
    [({}, True), ({"NO_COLOR": "1"}, False)],
    ids=["colors", "no-color"],
)
def test_terminal_logs_are_readable_lines(
    variables: dict[str, str], colored: bool
) -> None:
    script = textwrap.dedent(
        """
        import logging

        from langgraph_openai_serve.core.logging import begin_log_context
        from langgraph_openai_serve.server.logging import configure_logging

        configure_logging("my_app", root_level=logging.WARNING)
        begin_log_context("request-123")
        logging.getLogger("my_app.graphs").info("node.finished")
        """
    )
    # A pseudo-terminal makes stdout interactive, as when a person runs the command.
    primary, secondary = pty.openpty()
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
            [sys.executable, "-c", script],
            stdout=secondary,
            stderr=subprocess.PIPE,
            env={
                **{k: v for k, v in os.environ.items() if k != "NO_COLOR"},
                **variables,
            },
            check=False,
            text=True,
        )
        os.close(secondary)
        output = os.read(primary, 65536).decode()
    finally:
        os.close(primary)

    assert result.returncode == 0, result.stderr
    assert ("\x1b[" in output) is colored
    line = re.sub(r"\x1b\[[0-9;]*m", "", output).strip()
    assert "node.finished" in line
    assert "request_id=request-123" in line
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)
