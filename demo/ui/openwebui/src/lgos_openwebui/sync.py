"""Install the LGOS Functions into a running Open WebUI through its API."""

import argparse
import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import get_args

import httpx

from .functions.generic.gateway import GatewayType
from .functions.generic.pipe import Pipe
from .install import function_specs, install_functions


@dataclass(frozen=True)
class SyncSettings:
    """The gateway the Pipe uses and the Functions to install."""

    gateway_type: str
    gateway_url: str
    model_prefix: str
    mcp_connection_id: str
    function_id: str
    include_demo_functions: bool


async def admin_token(client: httpx.AsyncClient, environ: Mapping[str, str]) -> str:
    """Use an admin API key, or sign in where Open WebUI accepts passwords."""
    if api_key := environ.get("OPENWEBUI_API_KEY"):
        return api_key
    response = await client.post(
        "/api/v1/auths/signin",
        json={
            "email": environ["OPENWEBUI_ADMIN_EMAIL"],
            "password": environ["OPENWEBUI_ADMIN_PASSWORD"],
        },
    )
    if response.status_code == httpx.codes.FORBIDDEN:
        msg = "Open WebUI rejects password sign-in; set OPENWEBUI_API_KEY to an admin API key."
        raise ValueError(msg)
    session = response.raise_for_status().json()
    if "token" not in session:
        # MFA and accounts awaiting approval get a next sign-in step instead.
        msg = (
            "Open WebUI's password sign-in needs another step, such as MFA; "
            "set OPENWEBUI_API_KEY to an admin API key."
        )
        raise ValueError(msg)
    return session["token"]


async def sync(client: httpx.AsyncClient, settings: SyncSettings) -> dict[str, str]:
    """
    Install the Functions with an admin's ``client`` and set the Pipe's valves.

    The valves hold no credential: the Pipe authenticates like Open WebUI's own
    OpenAI connection to the gateway. Invalid settings fail before anything is
    written.
    """
    valves = Pipe.Valves.model_validate(
        {
            "OPENAI_GATEWAY_TYPE": settings.gateway_type,
            "OPENAI_GATEWAY_BASE_URL": settings.gateway_url,
            "MODEL_PREFIX": settings.model_prefix,
            "MCP_GATEWAY_CONNECTION_ID": settings.mcp_connection_id,
            # Without the example, its valve is kept like every valve not set here.
            **(
                {"USERVALVES_GRAPH_MODEL_ID": f"{settings.model_prefix}/simple-graph"}
                if settings.include_demo_functions
                else {}
            ),
        }
    )
    return await install_functions(
        client,
        function_specs(
            settings.function_id,
            valves=valves.model_dump(mode="json", exclude_unset=True),
            include_demo_functions=settings.include_demo_functions,
        ),
    )


async def _run(
    url: str, settings: SyncSettings, environ: Mapping[str, str]
) -> dict[str, str]:
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        client.headers["Authorization"] = f"Bearer {await admin_token(client, environ)}"
        return await sync(client, settings)


def main(argv: list[str] | None = None) -> None:
    """Run one operator-requested sync."""
    defaults = Pipe.Valves()
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "OPENWEBUI_API_KEY is an admin API key; without it, the command signs "
            "in with OPENWEBUI_ADMIN_EMAIL and OPENWEBUI_ADMIN_PASSWORD."
        ),
    )
    parser.add_argument("--url", required=True, help="Open WebUI root URL")
    parser.add_argument(
        "--gateway-type",
        required=True,
        choices=get_args(GatewayType),
        help="Gateway implementation",
    )
    parser.add_argument(
        "--gateway-url",
        required=True,
        help=(
            "Gateway root URL reachable from Open WebUI, without /v1; the Pipe "
            "authenticates like Open WebUI's OpenAI connection to <url>/v1"
        ),
    )
    parser.add_argument(
        "--model-prefix",
        default=defaults.MODEL_PREFIX,
        help="Gateway namespace of the graph models (default: %(default)s)",
    )
    parser.add_argument(
        "--mcp-connection-id",
        default=defaults.MCP_GATEWAY_CONNECTION_ID,
        help="Open WebUI MCP connection to the gateway (default: %(default)s)",
    )
    parser.add_argument(
        "--function-id",
        default="generic",
        help="Function ID; existing chats refer to it (default: %(default)s)",
    )
    parser.add_argument(
        "--include-demo-functions",
        action="store_true",
        help="Also install the UserValves example for <prefix>/simple-graph",
    )
    args = parser.parse_args(argv)
    environ = os.environ
    if not environ.get("OPENWEBUI_API_KEY") and not (
        environ.get("OPENWEBUI_ADMIN_EMAIL") and environ.get("OPENWEBUI_ADMIN_PASSWORD")
    ):
        parser.error(
            "set OPENWEBUI_API_KEY, or OPENWEBUI_ADMIN_EMAIL and OPENWEBUI_ADMIN_PASSWORD"
        )
    settings = SyncSettings(
        gateway_type=args.gateway_type,
        gateway_url=args.gateway_url,
        model_prefix=args.model_prefix,
        mcp_connection_id=args.mcp_connection_id,
        function_id=args.function_id,
        include_demo_functions=args.include_demo_functions,
    )
    try:
        results = asyncio.run(_run(args.url, settings, environ))
    except httpx.HTTPStatusError as exc:
        # Open WebUI explains a rejected Function, such as a failed import.
        msg = f"Open WebUI sync failed: {exc}\n{exc.response.text}"
        raise SystemExit(msg) from None
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        msg = f"Open WebUI sync failed: {exc}"
        raise SystemExit(msg) from None
    for function_id, action in results.items():
        print(f"{function_id}: {action}")


if __name__ == "__main__":
    main()
