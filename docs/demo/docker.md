# Docker Compose

## Self-Contained Demo Layout

The demo uses five independent uv projects rather than a uv workspace:

| Project | Lockfile | Deployment |
| --- | --- | --- |
| `demo/api` | `demo/api/uv.lock` | `ghcr.io/ilkersigirci/lgos-demo-api` |
| `demo/api-coding-agent` | `demo/api-coding-agent/uv.lock` | Local image build |
| `demo/files_api` | `demo/files_api/uv.lock` | `ghcr.io/ilkersigirci/lgos-files-api` |
| `demo/ui/chainlit_ui` | `demo/ui/chainlit_ui/uv.lock` | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `demo/ui/openwebui` | `demo/ui/openwebui/uv.lock` | Tests only; its source is mounted into the pinned `homeserver-open-webui` image |

Published project-owned images use only their project directories as build
contexts. The Compose entrypoint is `docker/compose/demo.yml`; service
definitions live in `docker/apps/`, while `docker/compose/development.yml`,
`docker/compose/otel.yml`, and `docker/compose/sso.yml` provide development,
OpenTelemetry, and SSO overlays.
Shared runtime assets remain under `demo/docker/`. The development overlay
additionally supplies the parent LGOS checkout as a named context for the demo API and coding-agent API
editable installs. Open WebUI runs the pinned `homeserver-open-webui` image,
the official slim image with small patches. Compose mounts the
`lgos_openwebui` package into it read-only and starts its
app factory, which installs the Functions at startup.
The two gateway fragments use pinned public images: upstream
Bifrost and `homeserver-litellm`. There is no demo-wide `pyproject.toml`, uv
workspace, shared Python environment, or shared lockfile.
The API package includes the compact Markdown corpus used by `lgos-rag`.

## Compose Modes

Commands use Bash and Just 1.58.0 or newer and run from the repository root.
In a standalone copy of `demo/`, use `just <recipe>` in place of
`just demo/<recipe>`.

!!! note "Docker Compose 5.3.0 or newer"

    The demo is validated with Compose 5.3.0.

Prepare the demo environment:

```bash
cp demo/.env.example demo/.env
```

Set `PUID` and `PGID` in `demo/.env` to the numeric host identity that owns the
bind directories. The checkout includes each empty service directory with a
tracked `.gitkeep`; service-created contents remain ignored.

`demo/.env.example` owns the configurable demo defaults. Compose reads the
corresponding values from `demo/.env` without supplying fallbacks. Keep an
existing `demo/.env` in sync with new template settings without overwriting
secrets.

Before using either OTEL mode, configure the [OpenTelemetry
settings](reference.md#opentelemetry-settings).

=== "Published images"

    `demo/docker/compose/demo.yml` uses published images for the demo API,
    Files API, and UIs. The coding-agent service builds locally:

    ```bash
    just demo/compose
    ```

    The command starts the gateway first, then the LGOS services, including
    Chainlit and DBHub, which assume it is running. It syncs metadata into the
    selected gateway, then starts Open WebUI. Services remain running in the
    background. Compose owns [dependency order and
    readiness](https://docs.docker.com/compose/how-tos/startup-order/); `just` only
    sequences the repeatable sync jobs.

    Set `DEMO_IMAGE_TAG` in `demo/.env` to select one
    release tag for the published project-owned demo images. To add the published OTEL
    overlay, use `just demo/compose --otel`.

=== "Build demo projects"

    Apply the explicit development model from the LGOS repository checkout.
    The demo API, coding-agent API, Files API, and Chainlit services build locally from their
    Dockerfiles and lockfiles; Open WebUI remains the pinned upstream image.
    Both graph API images install the parent LGOS
    checkout as an editable package:

    ```bash
    just demo/compose --dev
    ```

    To add the OTEL overlay while building the current checkout, use
    `just demo/compose --dev --otel`.

    The development overlay bind-mounts the Python application sources and the
    parent LGOS package read-only. Restart or recreate the affected service
    after source edits. Dependency metadata and lockfile changes require an
    image rebuild.

    ??? tip "Develop with the local Chainlit utilities"

        To try a sibling `../chainlit-utils` checkout before publishing, add
        the optional override:

        ```bash
        just demo/compose --dev --chainlit-utils
        ```

        This selects an editable build target and mounts the utility source
        read-only in the existing `lgos-chainlit` service. Only this mode
        requires the sibling checkout. Regular development uses the utility
        release recorded in the lockfile.

        After utility source edits, run `docker restart lgos-chainlit`.
        `just demo/up lgos-chainlit --dev` recreates the service without the
        override. Rerun the command above after dependency changes to rebuild.

=== "Test this LGOS checkout without containers"

    For immediate local feedback without containers, use uv's temporary
    editable overlay:

    ```bash
    just demo/test --editable
    ```

    This tests all five projects without changing their dependency declarations
    or lockfiles. Chainlit and Open WebUI remain standalone clients and exercise
    whichever API their OpenAI base URL targets.

### SSO Overlay

By default, Chainlit uses mock login, Open WebUI password login, and both send
`OPENAI_GATEWAY_API_KEY` for every user. Add `--sso` to any `just demo/compose`
command to switch the stack to OIDC:

```bash
just demo/compose --sso
```

The `docker/compose/sso.yml` overlay reads the SSO section of `demo/.env` and:

- signs users in to both UIs with OIDC, and makes each UI send the user's
  access token instead of the shared key;
- makes the bundled LiteLLM validate those RFC 9068 tokens with its native JWT
  auth and apply each user's models, budget, and spend. A user reaches the MCP
  servers marked `allow_all_keys` and those LiteLLM's Admin UI grants to them
  or to a team they belong to; a proxy admin gets LiteLLM's admin view. Other
  credentials, such as `OPENAI_GATEWAY_API_KEY`, `DEMO_GRAPH_GATEWAY_API_KEY`,
  and the sync's master key, keep LiteLLM's native authentication. Bifrost
  cannot validate the tokens, so `--sso` needs `OPENAI_GATEWAY_TYPE=litellm`;
- signs users in to LiteLLM's Admin UI through the same provider and as the
  same LiteLLM user. Only members of the provider's `admin` group become proxy
  admins and can open it; register `${DEMO_LITELLM_URL}/sso/callback` with
  LiteLLM's client.

A graph's own model calls use the graph APIs' `DEMO_GRAPH_GATEWAY_API_KEY` and
name the user in LiteLLM's `x-litellm-end-user-id` header. That key defaults to
the shared key, which the bundled LiteLLM uses as its master key, so create a
dedicated non-admin key in LiteLLM's Admin UI, such as one with the alias
`lgos-graphs`, and set it before exposing the stack. Open WebUI and Chainlit
send the token's subject as the OpenAI `safety_identifier`, and LGOS passes it to the graph's
model calls, so LiteLLM attributes both the graph request and its model calls
to that user.
LiteLLM's Logs group a session's rows per key, so a graph question shows the
user's row and the graph key's model-call rows. With the
[OpenTelemetry overlay](opentelemetry.md), both share one W3C trace ID as their
session; an admin clicking either row's Session ID sees the whole trace.

The overlay applies to the stack it starts; `just demo/up` recreates a service
without it. See [Chainlit login](chainlit.md#persistence-and-login) and
[Open WebUI gateway credentials](open-webui.md#gateway-credentials) for each
UI's settings.

## Demo Services

!!! note "One bundled gateway at a time"

    Bifrost and LiteLLM both publish host port `3000`. Enable only the selected
    gateway's profile, and stop the running gateway before switching. Changing
    `OPENAI_GATEWAY_TYPE` does not stop the previous gateway container.

=== "PostgreSQL MCP"

    ```bash
    just demo/up lgos-postgres-mcp --wait
    ```

    Once the demo API and Chainlit are healthy and have created their tables,
    the one-shot `lgos-mcp-db-setup` job creates or updates the dedicated `lgos_mcp`
    login and four curated views over live Chainlit users/conversations and
    pending LGOS interrupts. The role can read only those views. The pinned
    DBHub service then exposes six fixed reports only inside the Compose
    network; it publishes no host port. Starting either bundled gateway starts
    these dependencies, including Chainlit, automatically. Replace
    `LGOS_MCP_DB_PASSWORD` and `LGOS_MCP_AUTH_TOKEN` before startup. See
    [PostgreSQL Through Native MCP](graphs/mcp-postgres.md) for the complete
    client and security flow.

=== "Graph APIs"

    ```bash
    just demo/up lgos-demo-api
    just demo/up lgos-api-coding-agent
    ```

    Run each attached service in a separate terminal. Compose starts the shared
    PostgreSQL dependency automatically.

    - `lgos-demo-api`: `http://localhost:3004/v1`
    - `lgos-api-coding-agent`: gateway access via `lgos/coding-agent`

    For an independently deployed LiteLLM API, run the shared one-shot
    [model-sync job](litellm-sync.md) after the deployment's health check.

=== "Background Worker"

    The worker is an optional Compose profile. Configure a native
    `HATCHET_CLIENT_TOKEN`, enable background execution, and add the profile
    alongside the selected gateway:

    ```dotenv
    # Choose litellm or bifrost.
    OPENAI_GATEWAY_TYPE=litellm
    COMPOSE_PROFILES=${OPENAI_GATEWAY_TYPE},background
    LGOS_BACKGROUND=hatchet
    HATCHET_CLIENT_TOKEN=...
    ```

    Then start the ordered UI stack or only the worker and PostgreSQL:

    ```bash
    just demo/compose --dev
    # Or: just demo/up lgos-background-worker
    ```

    The `lgos-background-worker` process uses explicit Hatchet slots and the
    same PostgreSQL checkpointer, coordinator, and graph code as the APIs. It
    publishes no port; Compose checks Hatchet's worker health endpoint on port
    8001 inside the container. The API triggers the Hatchet task and reads its
    status and Response from Hatchet. See [Background Mock](graphs/background-mock.md).

=== "Files API"

    ```bash
    just demo/up lgos-files-api
    ```

    The independently packaged service connects directly to its configured
    S3-compatible store. It neither starts PostgreSQL nor imports LGOS.

    - central Files API: `http://localhost:3006/v1`

=== "Bifrost"

    ```bash
    just demo/compose
    ```

    The UIs use native `/openai/v1/responses`, normal `/v1` Files routing, the
    aggregate `/mcp` endpoint, and native `/v1/models` metadata. Compose writes
    zero-priced graph rows and synchronizes descriptions, features, and client
    settings automatically. Use `--dev` to build this checkout. The named
    PostgreSQL Virtual MCP remains available at
    `http://localhost:3000/mcp/lgos-postgres`. See [Bifrost Gateway](bifrost.md)
    for endpoints, routing, and the shared SDK verification command.

=== "External LiteLLM"

    Set these values in `demo/.env` to reuse an existing LiteLLM gateway:

    ```dotenv
    OPENAI_GATEWAY_TYPE=litellm
    COMPOSE_PROFILES=
    OPENAI_GATEWAY_BASE_URL=https://litellm.example.com
    DEMO_GATEWAY_HOST_URL=https://litellm.example.com
    OPENAI_GATEWAY_API_KEY=TO_BE_FILLED
    ```

    Use the gateway root without `/v1`. Both UIs share the gateway type, base
    URL, and `OPENAI_GATEWAY_API_KEY`, or send each user's token with the
    [SSO overlay](#sso-overlay). An external LiteLLM uses the same
    Responses, Files, and catalog routes as bundled LiteLLM.

    Both UIs derive their MCP endpoint from the same gateway root and use the
    same credential as Responses and Files. To use `mcp-postgres`, also adapt
    the checked-in
    `mcp_servers.lgos_postgres` entry and `LGOS_MCP_AUTH_TOKEN` for the external
    gateway, just as the Files configuration is adapted below. The external
    gateway URL must be reachable from both UI containers and from the graph
    API, worker, and coding-agent containers, which call their models through
    it. Set `DEMO_API_OPENAI_CHAT_COMPLETIONS_MODEL`,
    `DEMO_API_OPENAI_RESPONSES_MODEL`, `DEMO_API_OPENAI_EMBEDDING_MODEL`, and the
    `DEMO_AUDIO_*` models to IDs that gateway registers. The advanced graph's
    vector store also needs that gateway's `/openai_passthrough` route. On another host,
    expose DBHub only on a private address reachable by that gateway.

    `just demo/compose` (or the same command with `--dev`) starts
    the demo APIs, Files service, Chainlit, and DBHub, syncs both
    catalogs to the external LiteLLM, then starts Open WebUI. An empty
    `COMPOSE_PROFILES` disables both bundled gateways;
    the template normally selects one through
    `COMPOSE_PROFILES=${OPENAI_GATEWAY_TYPE}`. The external gateway and its
    administrator credentials must be reachable by the
    [model-sync job](litellm-sync.md).

    Enable LiteLLM's native database model storage. For Files, adapt the
    `files_settings` in [`docker/configs/litellm/config.yaml`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/docker/configs/litellm/config.yaml)
    to the shared Files service.

    On the same Docker host, attach the existing LiteLLM service to the demo's
    network through its own Compose deployment, alongside its current networks:

    ```yaml
    services:
      litellm:
        networks:
          # Retain the service's existing networks here too.
          - lgos

    networks:
      lgos:
        external: true
        name: lgos-network
    ```

    Create the network by starting the demo backends first, for example with
    `just demo/up lgos-demo-api`,
    `just demo/up lgos-api-coding-agent`, and
    `just demo/up lgos-files-api` in separate terminals, then
    `just demo/up lgos-postgres-mcp --wait`.
    The existing gateway can then resolve `lgos-demo-api`, `lgos-api-coding-agent`,
    `lgos-files-api`, and `lgos-postgres-mcp` using the bundled sync, Files, and
    MCP configuration examples. For another host,
    replace those upstream URLs with addresses reachable from that gateway.

    The external deployment continues to own its database, TLS, credentials,
    and Admin UI SSO. The selected credentials must allow the LGOS models, the
    `DEMO_API_OPENAI_*` model IDs the graphs call, Files operations, the
    configured MCP tools, and native `/model/info`. Chainlit can
    enable [delegated OAuth](chainlit.md#persistence-and-login); it then sends
    each user's token instead of the shared static key, MCP included.
    If the gateway already configures `litellm_proxy` Files,
    reconcile that provider with the demo's shared Files namespace. Then run
    the [LiteLLM SDK checks](#demo-services) against the external URL.

=== "LiteLLM"

    ```bash
    just demo/up lgos-litellm
    ```

    LiteLLM is one of the two first-class UI entry points. After startup,
    [sync all graph catalogs](litellm-sync.md#usage). The UIs use:

    - model metadata: `http://localhost:3000/model/info`
    - managed Files: `http://localhost:3000/v1`
    - managed routing: `http://localhost:3000/v1`
    - native MCP: `http://localhost:3000/mcp`
    - LiteLLM Admin UI: `http://localhost:3000/ui/`

    The bundled configuration uses API-key authentication. Chainlit can still
    use OAuth for browser login with that key. OAuth token forwarding additionally
    requires a gateway configured to validate delegated access tokens through its
    trusted SSO ingress; enabling forwarding in Chainlit alone does not configure
    LiteLLM. See [Chainlit OAuth](chainlit.md#persistence-and-login).

    Chainlit and Open WebUI send Responses and Files to managed routing and
    read descriptions, capabilities, and settings from `model_info.lgos`.
    Each graph uses LiteLLM's `model_name` unchanged for inference. The proxy retains
    normal model routing while each API remains the source of its descriptions
    and LGOS capability metadata and the Files service remains the owner of
    file bytes.
    Neither UI connects to an upstream service directly.
    `OPENAI_GATEWAY_API_KEY` protects these routes in the default setup; replace its demo-only
    default in any shared deployment. For the local Admin UI, sign in as
    `admin`; unless `UI_PASSWORD` is set separately, the password is the value
    of `OPENAI_GATEWAY_API_KEY` from `demo/.env`.

    The MCP configuration reaches DBHub with the separate internal
    `LGOS_MCP_AUTH_TOKEN`, allowlists the six fixed PostgreSQL reports, and
    permits authenticated gateway keys to use them. Native clients still opt
    in by connecting to `/mcp`; the graph API receives only ordinary OpenAI
    function definitions and results.

    The managed-routing surface uses concrete database-backed models
    and LiteLLM's native
    [Responses endpoint](https://docs.litellm.ai/docs/response_api). Select an
    API with a provider-qualified model, such as
    `lgos/custom-input-output-context`. The shared demo PostgreSQL service
    keeps LiteLLM's Admin UI and gateway-management records in its own
    `litellm` schema; graph execution state remains owned by LGOS. LiteLLM's
    standard `files_settings` route uses `provider=litellm_proxy` to isolate
    upload, retrieval, content, and deletion from the graph deployments.

    The LGOS-owned sync reads each API's model list and detail, then registers
    routing and `model_info.lgos` through LiteLLM's native management API.
    New deployments set `model_info.supports_native_streaming: true`.
    The default public
    [`homeserver-litellm` image](https://github.com/ilkersigirci/homeserver-docker/pkgs/container/homeserver-litellm)
    preserves native Responses streaming and polling-only background
    lifecycles. Compose
    reads its tag and digest from `DEMO_LITELLM_IMAGE` in `demo/.env`. Normal
    demo commands use this image without a local build or an image override.

    To use another compatible image, set `DEMO_LITELLM_IMAGE` in `demo/.env`
    or supply it on the command line:

    ```bash
    DEMO_LITELLM_IMAGE='registry.example.com/team/litellm:tag' \
      just demo/compose --dev --otel
    ```

    Keep the override set for subsequent Compose commands. To restore the
    default, copy the image value from `demo/.env.example`. An alternative image
    must preserve native Responses streaming and background IDs, authenticated
    `/model/info` with custom metadata, managed Files routing, and the Admin UI
    migration runtime.

    Managed routing also passes the tested Files lifecycle, file-ID input, and
    function continuation, while its rewritten standard error metadata remains
    a strict expected failure. The bundled image records successful managed
    Responses requests in LiteLLM's spend logs, including streaming requests.
    Token and spend values reflect usage and pricing supplied for the selected
    graph model.

    With the service healthy, run the focused OpenAI SDK check from the
    repository root. It tests managed routing, the catalog-to-inference
    flow, and native streaming fidelity against the direct LGOS test endpoints:

    ```bash
    just demo/test-litellm --editable
    ```

=== "Chainlit"

    With the gateway and its backends running:

    ```bash
    just demo/up lgos-chainlit
    ```

    Chainlit: `http://localhost:3002`

    This command starts Chainlit and PostgreSQL. Use
    `just demo/compose` to start the complete stack with the gateway
    selected by `COMPOSE_PROFILES`.

    Configure its signing secret as described in the
    [Chainlit client](chainlit.md).

=== "Open WebUI"

    ```bash
    just demo/up lgos-openwebui --wait
    ```

    Open WebUI: `http://localhost:3003`

    Compose runs the pinned `homeserver-open-webui` image. Its startup
    installs the bundled Functions, and its model list reads each graph's
    description and settings from the gateway; see the
    [Open WebUI setup](open-webui.md#setup). The image keeps chat uploads raw,
    as described under [Open WebUI file input](open-webui.md#file-input).

PostgreSQL is published on `localhost:3001`. LangGraph persistence, Bifrost
state, Open WebUI state—including its native raw file copies—and the
coding-agent workspace and Codex threads use host bind mounts under
`demo/docker/volumes/`; the Compose model declares no named volumes. Every
service runs as `PUID:PGID` with a read-only root filesystem, dropped
capabilities, and explicit resource limits. Narrow tmpfs mounts hold
required ephemeral writes. The graph APIs, background worker, and Chainlit
apply pending schema migrations when they start and stop if one fails. Each
takes a PostgreSQL advisory lock first, so processes that start together
migrate one at a time.

Chainlit stores thread and element metadata in PostgreSQL, while its native S3
client uploads generated file elements to the configured `BUCKET_NAME`.
Resuming a thread obtains a fresh signed object URL from that client. The
central Files API uses only its separate `DEMO_API_FILES_BUCKET`,
`DEMO_API_FILES_S3_ENDPOINT`, and `DEMO_API_FILES_AWS_*` settings. The two S3
configurations are independent.

The API and optional background worker share PostgreSQL for thread-scoped
application data, durable checkpoints, and fail-fast interrupt coordination.
Session-level
[advisory locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)
prevent two workers from advancing the same interrupt run at once; a contended
request fails instead of waiting. No Redis service is required. The lock is
held only while an API request or the background worker executes the graph,
never while a human is
deciding. A per-process capacity gate preserves a pool connection for
persistence I/O.

Compose also forces `LANGGRAPH_STRICT_MSGPACK=True` for the APIs. Strict
deserialization narrows which checkpoint object types LangGraph may
reconstruct, following its
[security guidance](https://github.com/langchain-ai/langgraph/security/advisories/GHSA-g48c-2wqr-h844).
Protect the PostgreSQL credentials and storage as integrity-sensitive data as
well.

!!! warning "PostgreSQL is sufficient state infrastructure, not a complete operations plan"

    The Compose database is a single demo container. A production deployment
    still owns tested [backup and restore](https://www.postgresql.org/docs/current/backup.html),
    monitoring, upgrades, and its chosen
    [replication and failover](https://www.postgresql.org/docs/current/high-availability.html)
    guarantees. LangGraph's exit durability writes the resumable state when an
    invocation pauses or finishes; LGOS drains that invocation before exposing
    interrupt tool calls. Whether the resulting commit survives loss of the
    primary depends on the PostgreSQL replication policy.

    Budget connections across every API replica and background worker. Each
    demo API or worker process has a five-connection pool and permits at most
    four simultaneous interrupt leases, preserving one connection for
    checkpoint I/O. Psycopg recommends
    monitoring pool statistics and sizing from observed workload; see its
    [pool guidance](https://www.psycopg.org/psycopg3/docs/advanced/pool.html#pool-connection-and-sizing).

    The coordinator uses session-level advisory locks and must retain one
    database session for the whole lease. If a proxy such as PgBouncer sits in
    front of PostgreSQL, use session pooling or a direct coordinator connection;
    PgBouncer documents session advisory locks as unsupported in
    [transaction-pooling mode](https://www.pgbouncer.org/features.html#sql-feature-map-for-pooling-modes).

!!! warning "Demo images are examples"

    The published images run the demo applications and graphs. They are not
    generic LGOS server images and should not be used as the base contract for
    an application that owns different graphs or dependencies.

Applications outside `demo/` own their container images and deployment model;
LGOS does not prescribe either. For exact demo commands and environment
variables, see [Demo Settings and Commands](reference.md).
