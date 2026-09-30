"""Configuration for the coding-agent service and its workspace."""

from pathlib import Path

from pydantic import DirectoryPath, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

GRAPH_ID = "coding-agent"
DESCRIPTION = (
    "Inspect and edit the workspace, run shell commands and tests, "
    "and explain the results using a coding agent (currently Codex)."
)


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEMO_CODING_AGENT_", extra="ignore")

    model: str = Field(min_length=1)
    base_url: str
    api_key: SecretStr
    workspace: DirectoryPath = Path("/workspace")
    timeout_seconds: float = Field(gt=0)
