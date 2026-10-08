"""Guard the files that keep the in-tree demo independently extractable."""

import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path
from string import Template

import anyio
import pytest
import yaml
from demo.ui.openwebui.src.lgos_openwebui.functions.generic.gateway import (
    MCP_GATEWAY_ID,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPOSITORY_ROOT / "demo"
REPOSITORY_BLOB_LINK = re.compile(
    r"https://github\.com/ilkersigirci/langgraph-openai-serve/blob/main/"
    r"(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)"
)


def test_openwebui_uses_the_pinned_upstream_image_without_a_custom_build() -> None:
    service = (DEMO_ROOT / "docker/apps/openwebui.yml").read_text(encoding="utf-8")

    assert re.search(
        r"^\s+image: ghcr\.io/open-webui/open-webui:\S+@sha256:[0-9a-f]{64}$",
        service,
        re.MULTILINE,
    )
    assert "build:" not in service
    assert not (DEMO_ROOT / "ui/openwebui/Dockerfile").exists()


def test_openwebui_mcp_connection_is_the_one_the_sync_attaches() -> None:
    compose = yaml.safe_load(
        (DEMO_ROOT / "docker/apps/openwebui.yml").read_text(encoding="utf-8")
    )
    environment = compose["services"]["lgos-openwebui"]["environment"]
    # Compose interpolates $VAR and ${VAR} like string.Template.
    connections = json.loads(
        Template(environment["TOOL_SERVER_CONNECTIONS"]).substitute(
            OPENAI_GATEWAY_BASE_URL="http://gateway:4000",
            OPENAI_GATEWAY_API_KEY="sk-demo",
        )
    )

    assert [
        (connection["url"], connection["key"], connection["info"]["id"])
        for connection in connections
    ] == [("http://gateway:4000/mcp", "sk-demo", MCP_GATEWAY_ID)]


@pytest.fixture
def task_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Record task arguments without starting project commands or services."""
    log = tmp_path / "commands.jsonl"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TASK_TEST_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], "
        "'cwd': os.getcwd(), "
        "'gateway_url': os.environ.get('OPENAI_GATEWAY_BASE_URL'), "
        "'gateway_key': os.environ.get('OPENAI_GATEWAY_API_KEY')}) + '\\n')\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("TASK_TEST_LOG", str(log))
    # Noninteractive SSH shells must also preserve the caller's executable PATH.
    monkeypatch.setenv("SSH_CLIENT", "127.0.0.1 50000 22")
    return log


async def test_demo_tests_forward_quoted_arguments_without_a_dotenv_file(
    tmp_path: Path, task_log: Path
) -> None:
    demo = tmp_path / "standalone demo"
    demo.mkdir()
    shutil.copyfile(DEMO_ROOT / "justfile", demo / "justfile")
    selection = "responses or (files and not slow)"

    result = await anyio.run_process(
        ["just", str(demo / "test"), "--", "-k", selection],
        env=os.environ,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    content = await anyio.Path(task_log).read_text(encoding="utf-8")
    commands = [json.loads(line) for line in content.splitlines()]
    assert [command["args"] for command in commands] == [
        ["run", "--directory", project, "--locked", "pytest", "-k", selection]
        for project in (
            "api",
            "files_api",
            "api-coding-agent",
            "ui/chainlit_ui",
            "ui/openwebui",
        )
    ]
    assert all(command["cwd"] == str(demo) for command in commands)


async def test_notebook_task_passes_host_literally(task_log: Path) -> None:
    host = "$(printf should-not-run)"

    result = await anyio.run_process(
        [
            "just",
            "--dotenv-path",
            str(DEMO_ROOT / ".env.example"),
            str(DEMO_ROOT / "marimo"),
            "--host",
            host,
            "--port",
            "2818",
        ],
        env=os.environ,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    command = json.loads(await anyio.Path(task_log).read_text(encoding="utf-8"))
    assert command["args"][-5:] == ["--host", host, "--port", "2818", "notebooks"]


@pytest.mark.parametrize("recipe", ["api", "background-worker", "marimo"])
async def test_local_graph_tasks_use_the_host_gateway(
    task_log: Path, monkeypatch: pytest.MonkeyPatch, recipe: str
) -> None:
    monkeypatch.setenv("DEMO_GATEWAY_HOST_URL", "http://localhost:4321")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "http://lgos-bifrost:4000")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "test-gateway-key")

    result = await anyio.run_process(
        [
            "just",
            "--dotenv-path",
            str(DEMO_ROOT / ".env.example"),
            str(DEMO_ROOT / recipe),
        ],
        env=os.environ,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    command = json.loads(await anyio.Path(task_log).read_text(encoding="utf-8"))
    assert command["gateway_url"] == "http://localhost:4321"
    assert command["gateway_key"] == "test-gateway-key"


def test_demo_api_lock_resolves_lgos_from_the_registry() -> None:
    lock = tomllib.loads((DEMO_ROOT / "api/uv.lock").read_text(encoding="utf-8"))
    lgos = next(
        package
        for package in lock["package"]
        if package["name"] == "langgraph-openai-serve"
    )

    assert lgos["source"] == {"registry": "https://pypi.org/simple"}


def test_files_api_has_no_graph_runtime_dependencies() -> None:
    project = tomllib.loads(
        (DEMO_ROOT / "files_api/pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads((DEMO_ROOT / "files_api/uv.lock").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    locked_packages = {package["name"] for package in lock["package"]}

    assert all(
        not dependency.startswith(("langgraph", "langchain"))
        for dependency in dependencies
    )
    assert not {
        package
        for package in locked_packages
        if package.startswith(("langgraph", "langchain"))
    }


def test_demo_api_does_not_own_file_storage() -> None:
    project = tomllib.loads(
        (DEMO_ROOT / "api/pyproject.toml").read_text(encoding="utf-8")
    )
    compose = (DEMO_ROOT / "docker/apps/demo-api.yml").read_text(encoding="utf-8")

    assert all(
        not dependency.startswith("boto3")
        for dependency in project["project"]["dependencies"]
    )
    for setting in (
        "DEMO_API_FILES_BUCKET",
        "DEMO_API_FILES_S3_ENDPOINT",
        "DEMO_API_FILES_AWS_ACCESS_KEY_ID",
        "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY",
        "DEMO_API_FILES_AWS_DEFAULT_REGION",
    ):
        assert setting not in compose


def test_bifrost_has_one_files_provider() -> None:
    config = json.loads(
        (DEMO_ROOT / "docker/configs/bifrost/config.json").read_text(encoding="utf-8")
    )
    file_requests = {
        "file_upload",
        "file_list",
        "file_retrieve",
        "file_delete",
        "file_content",
    }

    files_providers = {
        name
        for name, provider in config["providers"].items()
        if file_requests
        & provider.get("custom_provider_config", {}).get("allowed_requests", {}).keys()
    }

    assert files_providers == {"lgos-files"}
    assert config["providers"]["lgos-files"]["network_config"]["base_url"] == (
        "http://lgos-files-api:8000"
    )
    files_keys = config["providers"]["lgos-files"]["keys"]
    assert any(key.get("use_for_batch_api") is True for key in files_keys)


def test_bifrost_outwaits_a_coding_agent_request() -> None:
    env = (DEMO_ROOT / ".env.example").read_text(encoding="utf-8")
    [limit] = re.findall(
        r"^DEMO_CODING_AGENT_TIMEOUT_SECONDS=(\d+)$", env, re.MULTILINE
    )
    config = json.loads(
        (DEMO_ROOT / "docker/configs/bifrost/config.json").read_text(encoding="utf-8")
    )
    network = config["providers"]["lgos-api-coding-agent"]["network_config"]

    # Bifrost's request timeout bounds a whole non-streaming response. LGOS
    # keepalive comments reset its stream-idle timer.
    assert network["default_request_timeout_in_seconds"] > int(limit)


def test_bundled_gateways_serve_the_default_demo_models() -> None:
    env = dict(
        line.split("=", 1)
        for line in (DEMO_ROOT / ".env.example")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith(("DEMO_AUDIO_", "DEMO_API_OPENAI_"))
    )
    litellm = yaml.safe_load(
        (DEMO_ROOT / "docker/configs/litellm/config.yaml").read_text(encoding="utf-8")
    )
    deployments = {
        deployment["model_name"]: deployment["litellm_params"]
        for deployment in litellm["model_list"]
    }
    bifrost = json.loads(
        (DEMO_ROOT / "docker/configs/bifrost/config.json").read_text(encoding="utf-8")
    )
    [virtual_key] = bifrost["governance"]["virtual_keys"]
    grants = {
        grant["provider"]: grant["allowed_models"]
        for grant in virtual_key["provider_configs"]
    }

    for setting in (
        "DEMO_API_OPENAI_CHAT_COMPLETIONS_MODEL",
        "DEMO_API_OPENAI_RESPONSES_MODEL",
        "DEMO_API_OPENAI_EMBEDDING_MODEL",
        "DEMO_AUDIO_STT_MODEL",
        "DEMO_AUDIO_TTS_MODEL",
    ):
        model_id = env[setting]
        provider, _, model = model_id.partition("/")
        deployment = deployments[model_id]
        assert deployment["model"] == model_id
        assert deployment["api_base"] == "os.environ/OPENAI_UPSTREAM_API_BASE"
        assert deployment["api_key"] == "os.environ/OPENAI_UPSTREAM_API_KEY"
        assert model in grants[provider]


def test_files_and_chainlit_s3_are_independently_configured() -> None:
    files_compose = (DEMO_ROOT / "docker/apps/files-api.yml").read_text(
        encoding="utf-8"
    )
    chainlit_compose = (DEMO_ROOT / "docker/apps/chainlit.yml").read_text(
        encoding="utf-8"
    )

    for setting in (
        "DEMO_API_FILES_BUCKET",
        "DEMO_API_FILES_S3_ENDPOINT",
        "DEMO_API_FILES_AWS_ACCESS_KEY_ID",
        "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY",
        "DEMO_API_FILES_AWS_DEFAULT_REGION",
    ):
        assert re.search(rf"\b{setting}: \$\{{?{setting}\b", files_compose)
    assert "APP_AWS_" not in files_compose
    assert "DEV_AWS_ENDPOINT" not in files_compose
    assert re.search(r"\bBUCKET_NAME: \$\{?BUCKET_NAME\b", chainlit_compose)


def test_chainlit_receives_only_its_configuration() -> None:
    compose = (DEMO_ROOT / "docker/apps/chainlit.yml").read_text(encoding="utf-8")

    assert "env_file:" not in compose
    for setting in (
        "CHAINLIT_AUTH_SECRET",
        "OPENAI_GATEWAY_API_KEY",
        "DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING",
        "DEMO_CHAINLIT_LOGIN_TYPE",
        "DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS",
        "OAUTH_GENERIC_CLIENT_SECRET",
    ):
        assert re.search(rf"\b{setting}: \$\{{?{setting}\b", compose)
    for unrelated_secret in (
        "OPENAI_UPSTREAM_API_KEY",
        "DEMO_OPENWEBUI_ADMIN_PASSWORD",
        "LANGFUSE_SECRET_KEY",
        "LITELLM_MASTER_KEY",
    ):
        assert unrelated_secret not in compose


def test_compose_ci_supplies_both_independent_s3_configurations() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/demo-test.yml").read_text(
        encoding="utf-8"
    )

    for setting in (
        "DEMO_API_FILES_BUCKET",
        "DEMO_API_FILES_S3_ENDPOINT",
        "DEMO_API_FILES_AWS_ACCESS_KEY_ID",
        "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY",
        "DEMO_API_FILES_AWS_DEFAULT_REGION",
        "BUCKET_NAME",
        "APP_AWS_ACCESS_KEY",
        "APP_AWS_SECRET_KEY",
        "APP_AWS_REGION",
        "DEV_AWS_ENDPOINT",
    ):
        assert f"{setting}:" in workflow

    standalone_workflow = (DEMO_ROOT / ".github/workflows/test.yml").read_text(
        encoding="utf-8"
    )
    assert "cp .env.example .env" in standalone_workflow


def test_demo_repository_links_resolve() -> None:
    source_files = [
        DEMO_ROOT / "README.md",
        DEMO_ROOT / "api/README.md",
        DEMO_ROOT / "files_api/README.md",
        DEMO_ROOT / "api-coding-agent/README.md",
        DEMO_ROOT / "ui/chainlit_ui/README.md",
        DEMO_ROOT / "ui/openwebui/README.md",
    ]
    for source_root in (
        DEMO_ROOT / "api/src",
        DEMO_ROOT / "files_api/src",
        DEMO_ROOT / "api-coding-agent/src",
        DEMO_ROOT / "ui/chainlit_ui/src",
        DEMO_ROOT / "ui/openwebui/src",
    ):
        source_files.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix in {".md", ".py"}
        )

    for source_file in source_files:
        content = source_file.read_text(encoding="utf-8")
        for match in REPOSITORY_BLOB_LINK.finditer(content):
            linked_file = REPOSITORY_ROOT / match.group("path")
            assert linked_file.is_file(), (
                f"{source_file} links to missing {linked_file}"
            )
