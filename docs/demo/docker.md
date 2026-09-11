# Docker Compose

## Self-Contained Demo Layout

The demo uses four independent uv projects rather than a uv workspace:

| Project | Lockfile | Deployment |
| --- | --- | --- |
| `demo/api` | `demo/api/uv.lock` | `ghcr.io/ilkersigirci/lgos-demo-api` |
| `demo/files_api` | `demo/files_api/uv.lock` | `ghcr.io/ilkersigirci/lgos-files-api` |
| `demo/ui/chainlit_ui` | `demo/ui/chainlit_ui/uv.lock` | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `demo/ui/openwebui` | `demo/ui/openwebui/uv.lock` | Local Function sync command and upload-policy mount |

Published project-owned images use only their project directories as build
contexts. The Compose entrypoint is `docker/compose/demo.yml`; service
definitions live in `docker/apps/`, while `docker/compose/development.yml` and
`docker/compose/otel.yml` provide development and OpenTelemetry overlays.
Shared runtime assets remain under `demo/docker/`. The development overlay
additionally supplies the parent LGOS checkout as a named context for the API's
editable install. The Open WebUI
integration uses the official Open WebUI image and keeps its Function sync
command local. Compose also mounts its small raw-upload policy into that image;
it does not build a project-owned Open WebUI image. The two gateway fragments
use pinned public images: upstream Bifrost and `homeserver-litellm`. There is
no demo-wide `pyproject.toml`, uv workspace, shared Python environment, or
shared lockfile.
The API package includes the compact Markdown corpus used by `lgos-rag`.

## Compose Modes

!!! note "Docker Compose 5.3.0 or newer"

    Chainlit uses `pre_start` for its private schema migrations. The two API
    services instead share one dedicated `lgos-demo-api-setup` job and wait for
    its successful completion. This avoids running the same LangGraph
    checkpoint migration concurrently in both API workers.

Prepare the demo environment:

```bash
cd demo
cp .env.example .env
```

Set `PUID` and `PGID` in `.env` to the numeric host identity that owns the bind
directories. The checkout includes each empty service directory with a tracked
`.gitkeep`; service-created contents remain ignored.

`.env.example` owns the configurable demo defaults. Compose reads the
corresponding values from `.env` without supplying fallbacks. Keep an
existing `.env` in sync with new template settings without overwriting secrets.

Before using either OTEL mode, configure the [OpenTelemetry
settings](reference.md#opentelemetry-settings).

=== "Published images"

    `docker/compose/demo.yml` contains no local builds:

    ```bash
    make compose
    ```

    The command waits for the gateway and its dependencies, syncs LiteLLM when
    selected, waits for both UIs, and syncs Open WebUI. Services remain running
    in the background. Compose still owns [dependency order and
    readiness](https://docs.docker.com/compose/how-tos/startup-order/); Make only
    sequences the repeatable sync jobs.

    Set `DEMO_IMAGE_TAG` in `.env` to select one
    release tag for all project-owned demo images. To add the published OTEL
    overlay, use `make compose-otel`.

=== "Build demo projects"

    Apply the explicit development model from the LGOS repository checkout.
    The API, Files API, and Chainlit services build locally from their
    Dockerfiles and lockfiles. Only the API image installs the parent LGOS
    checkout as an editable package:

    ```bash
    make compose-dev
    ```

    To add the OTEL overlay while building the current checkout, use
    `make compose-otel-dev`.

    The development overlay bind-mounts the Python application sources and the
    parent LGOS package read-only. Restart or recreate the affected service
    after source edits. Dependency metadata and lockfile changes require an
    image rebuild.

=== "Test this LGOS checkout without containers"

    For immediate local feedback without containers, use uv's temporary
    editable overlay:

    ```bash
    uv run --directory api --locked --with-editable ../.. pytest
    ```

    This command does not rewrite `api/pyproject.toml` or `api/uv.lock`.
    Chainlit and Open WebUI remain standalone clients and exercise whichever API
    their OpenAI base URL targets.

## Demo Services

!!! note "One bundled gateway at a time"

    Bifrost and LiteLLM both publish host port `3000`. Enable only the selected
    gateway's profile, and stop the running gateway before switching. Changing
    `OPENAI_GATEWAY_TYPE` does not stop the previous gateway container.

=== "Graph APIs"

    ```bash
    make run-api-a
    make run-api-b
    ```

    Run each attached service in a separate terminal. Compose starts the shared
    PostgreSQL dependency automatically. Before either graph API starts,
    `lgos-demo-api-setup` waits for PostgreSQL health and initializes the
    LangGraph checkpoint and store schemas once. Both APIs use
    [`service_completed_successfully`](https://docs.docker.com/reference/compose-file/services/#depends_on)
    as their readiness dependency.

    - `lgos-a`: `http://localhost:3004/v1`
    - `lgos-b`: `http://localhost:3005/v1`

    For an independently deployed LiteLLM API, run the shared one-shot
    [model-sync job](litellm-sync.md) after the deployment's health check.

=== "Files API"

    ```bash
    make run-files
    ```

    The independently packaged service connects directly to its configured
    S3-compatible store. It neither starts PostgreSQL nor imports LGOS.

    - central Files API: `http://localhost:3006/v1`

=== "Bifrost"

    ```bash
    make run-bifrost
    ```

    The UIs use native `/openai/v1/responses`, normal `/v1` Files routing, and
    raw pass-through only for provider-specific catalog detail. See [Bifrost
    Gateway](bifrost.md) for endpoints, routing, and the shared SDK verification
    command.

=== "External LiteLLM"

    Set these values in `demo/.env` to reuse an existing LiteLLM gateway:

    ```dotenv
    OPENAI_GATEWAY_TYPE=litellm
    COMPOSE_PROFILES=
    OPENAI_GATEWAY_BASE_URL=https://litellm.example.com
    OPENAI_GATEWAY_API_KEY=TO_BE_FILLED
    ```

    Use the gateway root without `/v1`. Both UIs and the host-side Open WebUI
    sync command use these shared settings. An external LiteLLM uses the same
    Responses, Files, and catalog routes as bundled LiteLLM.

    `make compose` (or `make compose-dev`) starts the demo APIs, Files service,
    and PostgreSQL, syncs both catalogs to the external LiteLLM, then starts and
    syncs the UIs. An empty `COMPOSE_PROFILES` disables both bundled gateways;
    the template normally selects one through
    `COMPOSE_PROFILES=${OPENAI_GATEWAY_TYPE}`. The external gateway and its
    administrator credentials must be reachable by the
    [model-sync job](litellm-sync.md).

    Enable LiteLLM's native database model storage. For Files, adapt the
    `files_settings` in [`docker/configs/litellm/config.yaml`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/docker/configs/litellm/config.yaml)
    to the shared Files service. No LGOS catalog pass-through
    is required. Retain the bundled image's
    `LITELLM_ENABLE_RESPONSES_STREAMING_FIX=true` opt-in when using that image.

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
    `make run-api-a`, `make run-api-b`, and `make run-files` in separate
    terminals. The existing gateway can then resolve `lgos-demo-api-a`,
    `lgos-demo-api-b`, and `lgos-files-api` using the bundled sync examples
    and Files configuration. For another host, replace those upstream URLs
    with addresses reachable from that gateway.

    The external deployment continues to own its database, TLS, credentials,
    and Admin UI SSO. The selected credentials must allow the LGOS models, Files
    operations, and native `/model/info`. Chainlit can use
    [delegated OAuth](chainlit.md#persistence-and-login) without a shared key.
    If the gateway already configures `litellm_proxy` Files,
    reconcile that provider with the demo's shared Files namespace. Then run
    `make sync-openwebui` and the [LiteLLM SDK checks](#demo-services) against
    the external URL.

=== "LiteLLM"

    ```bash
    make run-litellm
    ```

    LiteLLM is one of the two first-class UI entry points. After startup,
    [sync both demo catalogs](litellm-sync.md#usage). The UIs use:

    - model metadata: `http://localhost:3000/model/info`
    - managed Files: `http://localhost:3000/v1`
    - managed routing: `http://localhost:3000/v1`
    - LiteLLM Admin UI: `http://localhost:3000/ui/`

    The bundled configuration uses API-key authentication. Chainlit's OAuth
    login requires a gateway configured to validate delegated access tokens
    through its trusted SSO ingress; enabling OAuth in Chainlit alone does
    not configure LiteLLM. See [Chainlit OAuth](chainlit.md#persistence-and-login).

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
    of `OPENAI_GATEWAY_API_KEY` from `.env`.

    The managed-routing surface uses concrete database-backed models
    and LiteLLM's native
    [Responses endpoint](https://docs.litellm.ai/docs/response_api). Select an
    API with a provider-qualified model, such as
    `lgos-a/custom-input-output-context` or
    `lgos-b/custom-input-output-context`. The shared demo PostgreSQL service
    keeps LiteLLM's Admin UI and gateway-management records in its own
    `litellm` schema; graph execution state remains owned by LGOS. LiteLLM's
    standard `files_settings` route uses `provider=litellm_proxy` to isolate
    upload, retrieval, content, and deletion from the graph deployments.

    The LGOS-owned sync reads each API's model list and detail, then registers
    routing and `model_info.lgos` through LiteLLM's native management API.
    New deployments set `model_info.supports_native_streaming: true`.
    The default public
    [`homeserver-litellm` image](https://github.com/ilkersigirci/homeserver-docker/pkgs/container/homeserver-litellm)
    preserves native Responses streaming. Compose
    reads its tag and digest from `DEMO_LITELLM_IMAGE` in `.env` and enables
    `LITELLM_ENABLE_RESPONSES_STREAMING_FIX=true` so it honors the deployment
    capability. Normal demo commands use this image without a local build or
    an image override.

    To use another compatible image, set `DEMO_LITELLM_IMAGE` in `demo/.env`
    or supply it on the command line from `demo/`:

    ```bash
    DEMO_LITELLM_IMAGE='registry.example.com/team/litellm:tag' \
      make compose-otel-dev
    ```

    Keep the override set for subsequent Compose commands. To restore the
    default, copy the image value from `.env.example`. An alternative image must preserve
    native Responses streaming, authenticated `/model/info` with custom metadata,
    managed Files routing, and the Admin UI migration runtime.

    Managed routing also passes the tested Files lifecycle, file-ID input, and
    function continuation, while its rewritten standard error metadata remains
    a strict expected failure. The bundled image records successful managed
    Responses requests in LiteLLM's spend logs, including streaming requests.
    Token and spend values reflect usage and pricing supplied for the selected
    graph model.

    With the service healthy, run the focused OpenAI SDK check from the
    repository root. It tests managed routing, the catalog-to-inference
    flow, and native streaming fidelity against the direct LGOS test endpoints.
    LiteLLM exposes no demo pass-through routes:

    ```bash
    make test-litellm
    ```

=== "Chainlit"

    With the gateway and its backends running:

    ```bash
    make run-chainlit
    ```

    Chainlit: `http://localhost:3002`

    This command starts Chainlit and PostgreSQL. Use `make compose` to start
    the complete stack with the gateway selected by `COMPOSE_PROFILES`.

    Configure its signing secret as described in the
    [Chainlit client](chainlit.md).

=== "Open WebUI"

    ```bash
    docker compose --env-file .env -f docker/compose/demo.yml up --wait lgos-openwebui
    ```

    Open WebUI: `http://localhost:3003`

    Compose runs the official Open WebUI image. Follow the
    [Open WebUI setup](open-webui.md#setup) to synchronize the bundled
    Functions and generate Workspace Models from LGOS metadata.
    The Compose service also mounts the temporary raw-upload policy described
    under [Open WebUI file input](open-webui.md#file-input).

PostgreSQL is published on `localhost:3001`. LangGraph persistence, Bifrost
state, and Open WebUI state—including its native raw file copies—use host bind
mounts under `demo/docker/volumes/`; the Compose model declares no named
volumes. Every service runs as `PUID:PGID` with a read-only root filesystem,
dropped capabilities, and explicit resource limits. Narrow tmpfs mounts hold
required ephemeral writes. The one-shot API setup service initializes the
LangGraph persistence schemas before both API workers, while Chainlit's
`pre_start` hook applies its independent UI migrations.

Chainlit stores thread and element metadata in PostgreSQL, while its native S3
client uploads generated file elements to the configured `BUCKET_NAME`.
Resuming a thread obtains a fresh signed object URL from that client. The
central Files API uses only its separate `DEMO_API_FILES_BUCKET`,
`DEMO_API_FILES_S3_ENDPOINT`, and `DEMO_API_FILES_AWS_*` settings. The two S3
configurations are independent.

The API workers share PostgreSQL for thread-scoped application data, durable
checkpoints, and fail-fast interrupt coordination. Session-level
[advisory locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)
prevent two workers from advancing the same interrupt run at once; a contended
request fails instead of waiting. No Redis service is required. The lock is
held only while an API request executes the graph, never while a human is
deciding. A per-process capacity gate preserves a pool connection for
persistence I/O.

Compose also forces `LANGGRAPH_STRICT_MSGPACK=true` for the APIs. Strict
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

    Budget connections across every API replica. Each demo API process has a
    five-connection pool and permits at most four simultaneous interrupt
    leases, preserving one connection for checkpoint I/O. Psycopg recommends
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
