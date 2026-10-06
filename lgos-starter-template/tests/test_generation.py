"""Validate generated projects at their distribution boundary."""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from cookiecutter.exceptions import FailedHookException
from cookiecutter.main import cookiecutter

TEMPLATE = Path(__file__).resolve().parents[1]


def test_generated_project_has_valid_code_and_configuration(tmp_path: Path) -> None:
    generated_project = Path(
        cookiecutter(str(TEMPLATE), no_input=True, output_dir=str(tmp_path))
    )
    for path in generated_project.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in generated_project.rglob("*.toml"):
        tomllib.loads(path.read_text(encoding="utf-8"))
    for pattern in ("*.yaml", "*.yml"):
        for path in generated_project.rglob(pattern):
            yaml.safe_load(path.read_text(encoding="utf-8"))
    check_python_style(generated_project)
    # Setup creates both; generation installs nothing.
    assert not (generated_project / ".env").exists()
    assert not (generated_project / "uv.lock").exists()


def check_python_style(project: Path) -> None:
    for arguments in (["check"], ["format", "--check"]):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *arguments, str(project)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.13", "3.14"])
def test_custom_identity_and_python_version(
    tmp_path: Path, python_version: str
) -> None:
    project = Path(
        cookiecutter(
            str(TEMPLATE),
            no_input=True,
            output_dir=str(tmp_path),
            extra_context={
                "project_name": "customer-support-assistant-service",
                "project_slug": "company_customer_support_assistant_with_document_search",
                "project_description": 'Research "agents" 🚀\nPath: C:\\workflows',
                "author": "İlker SIĞIRCI 🚀",
                "python_version": python_version,
                "license": "Proprietary",
            },
        )
    )
    config = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    package = "company_customer_support_assistant_with_document_search"
    assert (project / "src" / package / "registry.py").is_file()
    assert f"LGOS_REGISTRY={package}.registry:create_registry" in (
        project / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert (project / ".python-version").read_text(
        encoding="utf-8"
    ).strip() == python_version
    assert config["project"]["requires-python"] == f">={python_version},<3.15"
    assert (
        config["project"]["description"] == 'Research "agents" 🚀\nPath: C:\\workflows'
    )
    assert config["project"]["authors"][0]["name"] == "İlker SIĞIRCI 🚀"
    docs = tomllib.loads((project / "zensical.toml").read_text(encoding="utf-8"))
    assert docs["project"]["site_description"] == config["project"]["description"]
    assert not (project / "LICENSE").exists()
    check_python_style(project)


@pytest.mark.parametrize(
    "context",
    [
        {"project_name": "Invalid Name"},
        {"project_slug": "for"},
        {"project_slug": "invalid-name"},
    ],
)
def test_invalid_names_fail_before_generation(
    tmp_path: Path, context: dict[str, str]
) -> None:
    with pytest.raises(FailedHookException):
        cookiecutter(
            str(TEMPLATE),
            no_input=True,
            output_dir=str(tmp_path),
            extra_context=context,
        )
