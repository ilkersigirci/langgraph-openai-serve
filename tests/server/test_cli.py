import json
import os
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import] - Test-only subprocess with explicit arguments and no shell.
import sys
import time
from pathlib import Path

import httpx2

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_lgos_serve_runs_a_registry_from_the_working_directory() -> None:
    port = _free_port()
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("LGOS_")
    }
    environment |= {
        "LGOS_PORT": str(port),
        "LGOS_ENABLE_LANGFUSE": "False",
        # Container images name their registry once, through the environment.
        "LGOS_REGISTRY": "tests.server.support:create_registry",
    }
    process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        [str(Path(sys.executable).with_name("lgos")), "serve"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for an external process; there is no in-process event to await.
        deadline = time.monotonic() + 30
        while True:
            try:
                httpx2.get(f"http://127.0.0.1:{port}/v1/health").raise_for_status()
                break
            except httpx2.TransportError:
                assert process.poll() is None, process.communicate()[1]
                assert time.monotonic() < deadline, "lgos serve did not start"
                time.sleep(0.1)
        models = httpx2.get(f"http://127.0.0.1:{port}/v1/models").json()
    finally:
        process.terminate()
        stdout, _ = process.communicate(timeout=10)

    assert {model["id"] for model in models["data"]} == {"chat", "approval"}
    messages = {json.loads(line)["message"] for line in stdout.splitlines()}
    assert {"server.persistence.in_memory", "Application startup complete."} <= messages
