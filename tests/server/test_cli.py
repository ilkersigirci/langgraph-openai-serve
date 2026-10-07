import json
import os
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import] - Test-only subprocess with explicit arguments and no shell.
import sys
import time
from importlib.metadata import version
from pathlib import Path

import click
import httpx2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "tests.server.support:create_registry"
LGOS = str(Path(sys.executable).with_name("lgos"))


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _environment(**variables: str) -> dict[str, str]:
    # Server variables from the developer's shell must not reach the command.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LGOS_", "UVICORN_", "WEB_CONCURRENCY"))
    }
    return {**environment, "LGOS_ENABLE_LANGFUSE": "False", **variables}


def test_lgos_reports_the_installed_package_version() -> None:
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        [LGOS, "--version"], capture_output=True, check=True, text=True
    )

    assert result.stdout == f"lgos, version {version('langgraph_openai_serve')}\n"


def test_server_without_hatchet_background_skips_the_hatchet_sdk() -> None:
    # The SDK and gRPC add about a second to every process and test start.
    code = (
        "import sys, langgraph_openai_serve.server; print('hatchet_sdk' in sys.modules)"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        [sys.executable, "-c", code], capture_output=True, check=True, text=True
    )

    assert result.stdout == "False\n"


@pytest.mark.parametrize(
    ("arguments", "variables"),
    [
        # Container images name the registry and Uvicorn options in the environment.
        pytest.param([], {"LGOS_REGISTRY": REGISTRY}, id="environment"),
        # Platforms set WEB_CONCURRENCY, which Uvicorn reads for its worker count.
        pytest.param([REGISTRY], {"WEB_CONCURRENCY": "2"}, id="workers"),
    ],
)
def test_lgos_serve_runs_a_registry_from_the_working_directory(
    arguments: list[str], variables: dict[str, str]
) -> None:
    port = _free_port()
    process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        [LGOS, "serve", *arguments],
        cwd=PROJECT_ROOT,
        env=_environment(UVICORN_PORT=str(port), **variables),
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
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "lgos serve did not start"
                time.sleep(0.1)
        models = httpx2.get(f"http://127.0.0.1:{port}/v1/models").json()
    finally:
        process.terminate()
        stdout, _ = process.communicate(timeout=10)

    assert {model["id"] for model in models["data"]} == {"chat", "approval"}
    messages = {json.loads(line)["message"] for line in stdout.splitlines()}
    assert {"server.persistence.in_memory", "Application startup complete."} <= messages


@pytest.mark.parametrize(
    ("arguments", "variables", "env_file"),
    [
        pytest.param(
            [],
            {"LGOS_BACKGROUND": "memory", "WEB_CONCURRENCY": "2"},
            "",
            id="environment",
        ),
        pytest.param(["--workers", "2"], {}, "LGOS_BACKGROUND=memory\n", id="options"),
    ],
)
def test_lgos_serve_rejects_memory_background_across_workers(
    tmp_path: Path, arguments: list[str], variables: dict[str, str], env_file: str
) -> None:
    # Each worker keeps its own runs, so a poll can reach one that never saw it.
    settings = tmp_path / ".env"
    settings.write_text(env_file)
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Arguments come only from this test module.
        [LGOS, "serve", REGISTRY, "--env-file", str(settings), *arguments],
        cwd=PROJECT_ROOT,
        env=_environment(UVICORN_PORT=str(_free_port()), **variables),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == click.UsageError.exit_code, result.stdout
    assert "LGOS_BACKGROUND=memory" in result.stderr
