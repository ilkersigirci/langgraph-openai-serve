"""Production logging configuration tests."""

import json
import subprocess
import sys
import textwrap


def test_json_logging_preserves_lgos_context_fields() -> None:
    script = textwrap.dedent(
        """
        import logging

        from lgos_demo_api.logging import configure_logging

        configure_logging()
        logging.getLogger("langgraph_openai_serve.test").info(
            "test.event",
            extra={
                "request_id": "request-123",
                "model": "simple-graph",
                "stream": False,
                "color_message": "stale ANSI message",
            },
        )
        logging.getLogger("uvicorn.error").warning("server.event")

        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logging.getLogger("langgraph_openai_serve.test").exception(
                "http.request.failed",
                extra={"request_id": "request-456"},
            )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    record, server_record, error_record = map(json.loads, result.stdout.splitlines())
    assert record["level"] == "info"
    assert record["logger"] == "langgraph_openai_serve.test"
    assert record["message"] == "test.event"
    assert record["request_id"] == "request-123"
    assert record["model"] == "simple-graph"
    assert record["stream"] is False
    assert record["timestamp"]
    assert "color_message" not in record
    assert server_record["logger"] == "uvicorn.error"
    assert server_record["message"] == "server.event"
    assert error_record["level"] == "error"
    assert error_record["message"] == "http.request.failed"
    assert error_record["request_id"] == "request-456"
    assert "RuntimeError: boom" in error_record["exception"]
