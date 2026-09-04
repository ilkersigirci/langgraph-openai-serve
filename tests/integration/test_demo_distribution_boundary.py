"""Guard the files that make the in-tree demo independently extractable."""

import json
import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPOSITORY_ROOT / "demo"
REPOSITORY_BLOB_LINK = re.compile(
    r"https://github\.com/ilkersigirci/langgraph-openai-serve/blob/main/"
    r"(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)"
)


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
        & provider["custom_provider_config"].get("allowed_requests", {}).keys()
    }

    assert files_providers == {"lgos-files"}
    assert config["providers"]["lgos-files"]["network_config"]["base_url"] == (
        "http://lgos-files-api:8000"
    )
    files_keys = config["providers"]["lgos-files"]["keys"]
    assert any(key.get("use_for_batch_api") is True for key in files_keys)


def test_files_and_chainlit_s3_are_independently_configured() -> None:
    files_compose = (DEMO_ROOT / "docker/apps/files-api.yml").read_text(
        encoding="utf-8"
    )
    chainlit_compose = (DEMO_ROOT / "docker/apps/chainlit.yml").read_text(
        encoding="utf-8"
    )

    assert "DEMO_API_FILES_BUCKET: ${DEMO_API_FILES_BUCKET:" in files_compose
    assert "DEMO_API_FILES_S3_ENDPOINT: ${DEMO_API_FILES_S3_ENDPOINT:" in files_compose
    assert (
        "DEMO_API_FILES_AWS_ACCESS_KEY_ID: ${DEMO_API_FILES_AWS_ACCESS_KEY_ID:"
        in files_compose
    )
    assert (
        "DEMO_API_FILES_AWS_SECRET_ACCESS_KEY: "
        "${DEMO_API_FILES_AWS_SECRET_ACCESS_KEY:" in files_compose
    )
    assert (
        "DEMO_API_FILES_AWS_DEFAULT_REGION: "
        "${DEMO_API_FILES_AWS_DEFAULT_REGION:" in files_compose
    )
    assert "${APP_AWS_" not in files_compose
    assert "${DEV_AWS_ENDPOINT" not in files_compose
    assert "BUCKET_NAME: ${BUCKET_NAME:" in chainlit_compose


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


def test_root_workflows_delegate_shared_steps_to_demo_actions() -> None:
    workflow_actions = {
        "demo-image-api.yml": "build-image",
        "demo-image-files-api.yml": "build-image",
        "demo-image-chainlit.yml": "build-image",
        "demo-test.yml": "check-project",
    }

    for workflow_name, action_name in workflow_actions.items():
        action = DEMO_ROOT / ".github/actions" / action_name / "action.yml"
        workflow = REPOSITORY_ROOT / ".github/workflows" / workflow_name

        assert action.is_file()
        assert f"uses: ./demo/.github/actions/{action_name}" in workflow.read_text(
            encoding="utf-8"
        )


def test_standalone_workflows_use_the_same_demo_actions() -> None:
    workflow_actions = {
        "image-api.yml": "build-image",
        "image-files-api.yml": "build-image",
        "image-chainlit.yml": "build-image",
        "test.yml": "check-project",
    }

    for workflow_name, action_name in workflow_actions.items():
        workflow = DEMO_ROOT / ".github/workflows" / workflow_name

        assert f"uses: ./.github/actions/{action_name}" in workflow.read_text(
            encoding="utf-8"
        )


def test_demo_repository_links_resolve() -> None:
    source_files = [
        DEMO_ROOT / "README.md",
        DEMO_ROOT / "api/README.md",
        DEMO_ROOT / "files_api/README.md",
        DEMO_ROOT / "ui/chainlit_ui/README.md",
        DEMO_ROOT / "ui/openwebui/README.md",
    ]
    for source_root in (
        DEMO_ROOT / "api/src",
        DEMO_ROOT / "files_api/src",
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
