from pathlib import Path

import pytest

from lgos_chainlit.lgos_protocol import ModelClientSettings


@pytest.fixture(autouse=True)
def chainlit_app_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run the Chainlit test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
def runtime_client_settings() -> ModelClientSettings:
    return ModelClientSettings.model_validate(
        {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "use_history": {
                        "type": "boolean",
                        "title": "Use conversation history",
                        "default": True,
                    },
                    "mode": {
                        "type": "string",
                        "title": "Mode",
                        "enum": ["brief", "detailed"],
                        "default": "brief",
                    },
                    "assistant_name": {
                        "type": "string",
                        "title": "Assistant name",
                        "minLength": 1,
                        "default": "Helper",
                    },
                },
            },
            "defaults": {
                "use_history": True,
                "mode": "brief",
                "assistant_name": "Helper",
            },
        }
    )
