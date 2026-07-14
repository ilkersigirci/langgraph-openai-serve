"""Inspect LGOS model fields through Bifrost's OpenAI pass-through route.

Example:
    uv run --env-file .env python scripts/check_bifrost_lgos_models.py | jq
"""

import json
import os

from openai import OpenAI

BIFROST_MODEL_PROVIDER = "lgos"


def main() -> None:
    client = OpenAI(
        base_url=os.environ["DEMO_OPENAI_BASE_URL"],
        api_key=os.environ["DEMO_OPENAI_API_KEY"],
        default_headers={"x-model-provider": BIFROST_MODEL_PROVIDER},
    )

    models = client.models.list()
    extension_found = False

    for model in models.data:
        fields = model.model_dump(mode="json")
        extension_found |= "langgraph_openai_serve" in fields
        print(json.dumps(fields, indent=2))

    if not extension_found:
        raise SystemExit(
            "No langgraph_openai_serve extension found. "
            "Check the pass-through URL and x-model-provider value."
        )


if __name__ == "__main__":
    main()
