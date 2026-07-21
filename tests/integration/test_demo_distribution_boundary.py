"""Guard the files that make the in-tree demo independently extractable."""

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


def test_root_workflows_delegate_shared_steps_to_demo_actions() -> None:
    workflow_actions = {
        "demo-image-api.yml": "build-image",
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
        DEMO_ROOT / "ui/chainlit_ui/README.md",
        DEMO_ROOT / "ui/openwebui/README.md",
    ]
    for source_root in (
        DEMO_ROOT / "api/src",
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
