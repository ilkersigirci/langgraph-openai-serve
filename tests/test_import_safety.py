"""Regression coverage for package import configuration sources.

This test uses an isolated subprocess to ensure that merely importing the
package does not accidentally load a `.env` file from the consumer's current
working directory (e.g., if `env_file=".env"` is ever accidentally re-added
to the package's SettingsConfigDict).
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_ENVIRONMENT_VARIABLES = (
    "LGOS_OPENAI_API_PREFIX",
    "LGOS_OPENAI_API_DOCS_ENABLED",
    "LGOS_ENABLE_LANGFUSE",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


def _import_settings(
    working_directory: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for variable in SETTINGS_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    if extra_env:
        environment.update(extra_env)

    source_directory = str(PROJECT_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_directory, existing_pythonpath) if part
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from langgraph_openai_serve.core.settings import settings; "
                "print(settings.OPENAI_API_PREFIX, "
                "settings.OPENAI_API_DOCS_ENABLED, "
                "settings.ENABLE_LANGFUSE)"
            ),
        ],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_import_ignores_working_directory_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "LGOS_OPENAI_API_PREFIX=not-a-path\nLGOS_OPENAI_API_DOCS_ENABLED=not-a-boolean\nLGOS_ENABLE_LANGFUSE=true\nLANGFUSE_BASE_URL=not-a-url",
        encoding="utf-8",
    )

    result = _import_settings(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/v1 False False"


def test_package_settings_still_read_process_environment(tmp_path: Path) -> None:
    result = _import_settings(
        tmp_path,
        extra_env={
            "LGOS_OPENAI_API_PREFIX": "/openai/v1/",
            "LGOS_OPENAI_API_DOCS_ENABLED": "true",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/openai/v1 True False"
