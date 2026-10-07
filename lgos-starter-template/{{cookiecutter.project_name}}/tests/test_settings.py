import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from {{ cookiecutter.project_slug }}.settings import (
    Settings,
)

ENV_EXAMPLE = Path(__file__).parents[1] / ".env.example"


def test_env_example_lacks_only_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # `just setup` copies the example, so it must define every other setting.
    for name in [name for name in os.environ if name.startswith("APP_")]:
        monkeypatch.delenv(name)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=ENV_EXAMPLE)
    assert [item["loc"] for item in error.value.errors()] == [("APP_OPENAI_API_KEY",)]
