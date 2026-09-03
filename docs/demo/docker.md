# Docker Compose

## Self-Contained Demo Layout

The demo uses four independent uv projects rather than a uv workspace:

| Project | Lockfile | Deployment |
| --- | --- | --- |
| `demo/api` | `demo/api/uv.lock` | `ghcr.io/ilkersigirci/lgos-demo-api` |
| `demo/files_api` | `demo/files_api/uv.lock` | `ghcr.io/ilkersigirci/lgos-files-api` |
| `demo/ui/chainlit_ui` | `demo/ui/chainlit_ui/uv.lock` | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `demo/ui/openwebui` | `demo/ui/openwebui/uv.lock` | Local Function sync command |

Published project-owned images use only their project directories as build
contexts. The Compose entrypoint is `docker/compose/demo.yml`; service
definitions live in `docker/apps/`, while `docker/compose/development.yml` and
`docker/compose/otel.yml` provide development and OpenTelemetry overlays.
Shared runtime assets remain under `demo/docker/`. The development overlay
additionally supplies the parent LGOS checkout as a named context for the API's
editable install. The Open WebUI
integration uses the official Open WebUI image and keeps its Function sync
command local. There is no demo-wide `pyproject.toml`, uv workspace, shared
Python environment, or shared lockfile. The API package includes the compact
Markdown corpus used by `lgos-rag`.

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

Before using either OTEL mode, configure the [OpenTelemetry
settings](reference.md#opentelemetry-settings).

=== "Published images"

    `docker/compose/demo.yml` contains no local builds:

    ```bash
    make compose
    ```

    `DEMO_IMAGE_TAG` defaults to `latest`. Set it in `.env` to select one
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

=== "Graph APIs"

    ```bash
    make run-api
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

=== "Files API"

    ```bash
    make run-files
    ```

    The independently packaged service connects directly to its configured
    S3-compatible store. It neither starts PostgreSQL nor imports LGOS.

    - central Files API: `http://localhost:3006/v1`

=== "Bifrost"

    ```bash
    docker compose -f docker/compose/demo.yml up --wait lgos-bifrost
    ```

    See [Bifrost Gateway](bifrost.md) for endpoints, routing, and the shared SDK
    verification command.

=== "Chainlit"

    ```bash
    make run-chainlit
    ```

    Chainlit: `http://localhost:3002`

    Configure its signing secret as described in the
    [Chainlit client](chainlit.md).

=== "Open WebUI"

    ```bash
    docker compose -f docker/compose/demo.yml up --wait lgos-openwebui
    make sync-openwebui
    ```

    Open WebUI: `http://localhost:3003`

    Compose runs the official Open WebUI image. The local sync command updates
    the bundled Functions and generates Workspace Models from LGOS metadata. See the
    [Open WebUI Functions](open-webui.md).

PostgreSQL is published on `localhost:3001`. LangGraph persistence, Bifrost
state, and Open WebUI state use host bind mounts
under `demo/docker/volumes/`; the Compose model declares no named volumes. Every
service runs as `PUID:PGID` with a read-only root filesystem, dropped
capabilities, and explicit resource limits. Narrow tmpfs mounts hold required
ephemeral writes. The one-shot API setup service initializes the LangGraph
persistence schemas before both API workers, while Chainlit's `pre_start` hook
applies its independent UI migrations.

Chainlit stores thread and element metadata in PostgreSQL, while its native S3
client uploads Plotly figure JSON to the configured `BUCKET_NAME`. Resuming a
thread obtains a fresh signed object URL from that client. The central Files API
uses only its separate `DEMO_API_FILES_BUCKET`, `DEMO_API_FILES_S3_ENDPOINT`,
and `DEMO_API_FILES_AWS_*` settings. The two S3 configurations are independent.

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
