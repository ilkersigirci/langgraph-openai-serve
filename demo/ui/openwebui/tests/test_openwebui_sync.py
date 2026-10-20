"""Installing the Functions into a running Open WebUI through its API."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from lgos_openwebui import sync as sync_command
from lgos_openwebui.sync import SyncSettings, admin_token, sync
from tests.support import ADMIN_TOKEN, FakeOpenWebUI

GATEWAY_URL = "http://gateway.internal:4000"


def deployment(**values) -> SyncSettings:
    return SyncSettings(
        **{
            "gateway_type": "litellm",
            "gateway_url": GATEWAY_URL,
            "model_prefix": "acme",
            "mcp_connection_id": "acme-tools",
            "function_id": "acme_pipe",
            "include_demo_functions": False,
            **values,
        }
    )


async def test_sync_installs_the_pipe_with_its_gateway_but_no_credential():
    unrelated = {"id": "unrelated", "name": "Keep", "content": "external"}
    app = FakeOpenWebUI([deepcopy(unrelated)])
    async with app.client() as client:
        assert await sync(client, deployment()) == {"acme_pipe": "created"}
        app.writes.clear()
        assert await sync(client, deployment()) == {"acme_pipe": "unchanged"}
        assert app.writes == []

    assert app.functions["unrelated"] == unrelated
    assert app.functions["acme_pipe"]["is_active"] is True
    # The Pipe authenticates like Open WebUI's own connection to the gateway.
    assert app.functions["acme_pipe"]["valves"] == {
        "OPENAI_GATEWAY_TYPE": "litellm",
        "OPENAI_GATEWAY_BASE_URL": GATEWAY_URL,
        "MODEL_PREFIX": "acme",
        "MCP_GATEWAY_CONNECTION_ID": "acme-tools",
    }


async def test_demo_functions_add_the_uservalves_example():
    app = FakeOpenWebUI([])
    async with app.client() as client:
        await sync(client, deployment(include_demo_functions=True))
        # A later sync without the example keeps it, as it keeps other valves.
        await sync(client, deployment())

    assert app.functions["uservalves_simple"]["is_active"] is True
    valves = app.functions["acme_pipe"]["valves"]
    assert valves["USERVALVES_GRAPH_MODEL_ID"] == "acme/simple-graph"


@pytest.mark.parametrize(
    "values",
    [{"model_prefix": "acme/graphs"}, {"gateway_url": "not-a-url"}],
    ids=["invalid-prefix", "invalid-url"],
)
async def test_invalid_settings_write_nothing(values: dict):
    app = FakeOpenWebUI([])
    async with app.client() as client:
        with pytest.raises(ValidationError):
            await sync(client, deployment(**values))
    assert app.writes == []


@pytest.mark.parametrize(
    ("environ", "token", "signins"),
    [
        (
            {
                "OPENWEBUI_API_KEY": "sk-admin",
                "OPENWEBUI_ADMIN_EMAIL": "admin@example.com",
                "OPENWEBUI_ADMIN_PASSWORD": "wrong-password",
            },
            "sk-admin",
            0,
        ),
        (
            {
                "OPENWEBUI_ADMIN_EMAIL": "admin@example.com",
                "OPENWEBUI_ADMIN_PASSWORD": "password",
            },
            ADMIN_TOKEN,
            1,
        ),
    ],
    ids=["api-key", "password"],
)
async def test_admin_token_prefers_an_api_key_to_password_sign_in(
    environ: dict[str, str], token: str, signins: int
):
    app = FakeOpenWebUI([])
    async with app.client() as client:
        assert await admin_token(client, environ) == token
    assert app.signins == signins


@pytest.mark.parametrize(
    "signin",
    # The SSO overlay turns password sign-in off; MFA adds a sign-in step.
    [{"password_auth": False}, {"signin_step": "mfa"}],
    ids=["password-sign-in-off", "another-sign-in-step"],
)
async def test_admins_without_a_password_token_need_an_api_key(signin: dict):
    app = FakeOpenWebUI([], **signin)
    async with app.client() as client:
        with pytest.raises(ValueError, match="OPENWEBUI_API_KEY"):
            await admin_token(
                client,
                {
                    "OPENWEBUI_ADMIN_EMAIL": "admin@example.com",
                    "OPENWEBUI_ADMIN_PASSWORD": "password",
                },
            )


def test_cli_reports_why_open_webui_rejected_a_function(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENWEBUI_API_KEY", ADMIN_TOKEN)
    app = FakeOpenWebUI([], rejection="No module named 'plotly'")

    async def run(url, settings, environ):
        async with app.client() as client:
            return await sync(client, settings)

    monkeypatch.setattr(sync_command, "_run", run)
    with pytest.raises(SystemExit) as exc:
        sync_command.main(
            [
                "--url=https://remote-webui.example",
                "--gateway-type=litellm",
                f"--gateway-url={GATEWAY_URL}",
            ]
        )

    assert "No module named 'plotly'" in str(exc.value)


def test_cli_rejects_an_unknown_gateway_type_before_contacting_open_webui(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("OPENWEBUI_API_KEY", ADMIN_TOKEN)

    with pytest.raises(SystemExit) as exc:
        sync_command.main(
            [
                "--url=https://remote-webui.invalid",
                "--gateway-type=other",
                f"--gateway-url={GATEWAY_URL}",
            ]
        )

    assert exc.value.code == 2
    assert "invalid choice: 'other'" in capsys.readouterr().err
