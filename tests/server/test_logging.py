import json
import subprocess  # ruff: ignore[suspicious-subprocess-import] - Test-only subprocess with explicit arguments and no shell.
import sys
import textwrap


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
