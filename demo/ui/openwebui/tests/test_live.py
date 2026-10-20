"""Opt-in checks of the sync and the Pipe against the running demo Open WebUI."""

import os
import sys

import anyio
import httpx
import pytest

from lgos_openwebui.sync import admin_token

pytestmark = pytest.mark.integration


async def demo_sync() -> dict[str, str]:
    """Run the demo's sync command; return each Function's action."""
    result = await anyio.run_process(
        [
            sys.executable,
            "-m",
            "lgos_openwebui.sync",
            f"--url={os.environ['DEMO_OPENWEBUI_HOST_URL']}",
            f"--gateway-type={os.environ['OPENAI_GATEWAY_TYPE']}",
            f"--gateway-url={os.environ['OPENAI_GATEWAY_BASE_URL']}",
            "--include-demo-functions",
        ],
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    return dict(line.split(": ", 1) for line in result.stdout.decode().splitlines())


async def test_a_repeated_sync_writes_nothing() -> None:
    await demo_sync()

    assert await demo_sync() == {
        "generic": "unchanged",
        "uservalves_simple": "unchanged",
    }


async def test_only_the_pipe_lists_the_graphs_without_stored_model_records() -> None:
    await demo_sync()
    async with httpx.AsyncClient(
        base_url=os.environ["DEMO_OPENWEBUI_HOST_URL"], timeout=30
    ) as client:
        client.headers["Authorization"] = (
            f"Bearer {await admin_token(client, os.environ)}"
        )
        config = (await client.get("/openai/config")).raise_for_status().json()
        if config["OPENAI_API_CONFIGS"]["0"]["auth_type"] == "system_oauth":
            pytest.skip("An admin API key carries no user token for the Pipe to send.")
        models = (await client.get("/api/models")).raise_for_status().json()["data"]
        # Export leaves out the records that override a listed model's info.
        records = (await client.get("/api/v1/models/base")).raise_for_status().json()

    model_ids = {model["id"] for model in models}
    graphs = {
        model_id for model_id in model_ids if model_id.startswith("generic.lgos/")
    }
    assert graphs
    # The connection's exclude_model_ids leaves the Pipe's namespace out.
    assert not {model_id for model_id in model_ids if model_id.startswith("lgos/")}
    # A stored record would replace the model info the Pipe lists.
    assert not graphs & {record["id"] for record in records}
