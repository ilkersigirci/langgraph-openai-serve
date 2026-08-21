# Docker Compose

## Self-Contained Demo Layout

The demo uses three independent uv projects rather than a uv workspace:

| Project | Lockfile | Deployment |
| --- | --- | --- |
| `demo/api` | `demo/api/uv.lock` | `ghcr.io/ilkersigirci/lgos-demo-api` |
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
    release tag for both project-owned demo images. To add the published OTEL
    overlay, use `make compose-otel`.

=== "Build demo projects"

    Apply the explicit development model from the LGOS repository checkout.
    The API and Chainlit services build locally from their Dockerfiles and
    lockfiles. The API image installs the parent LGOS checkout as an editable
    package:

    ```bash
    make compose-dev
    ```

    To add the OTEL overlay while building the current checkout, use
    `make compose-otel-dev`.

    The development overlay bind-mounts the demo API source and parent LGOS
    package read-only. Restart or recreate the affected service after source
    edits. Both packages are installed editable in the development image;
    dependency metadata and lockfile changes require an image rebuild.

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

=== "APIs"

    ```bash
    make run-api
    make run-api-b
    ```

    Run each attached service in a separate terminal. Compose starts their
    shared PostgreSQL dependency automatically. Before either API starts,
    `lgos-demo-api-setup` waits for PostgreSQL health and initializes the
    checkpoint schema once. Both APIs use
    [`service_completed_successfully`](https://docs.docker.com/reference/compose-file/services/#depends_on)
    as their readiness dependency.

    - `lgos-a`: `http://localhost:3004/v1`
    - `lgos-b`: `http://localhost:3005/v1`

=== "Bifrost"

    ```bash
    docker compose -f docker/compose/demo.yml up --wait lgos-bifrost
    ```

    Use `http://localhost:3000/v1` as the provider-qualified model catalog. Use
    `http://localhost:3000/openai_passthrough/v1` for detailed model retrieval
    and inference, sending the selected model's provider prefix as
    `x-model-provider`. Both dynamic UI integrations discover this routing
    information from the catalog. Chainlit uses direct mode only when its
    optional catalog URL is unset.

    From the package repository, run `make test-bifrost` to verify both APIs,
    detailed model metadata, inference, and client events through one SDK
    client. See
    [Bifrost Gateway](bifrost.md).

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

PostgreSQL is published on `localhost:3001`. PostgreSQL checkpoints, Bifrost
state, and Open WebUI state use host bind mounts
under `demo/docker/volumes/`; the Compose model declares no named volumes. Every
service runs as `PUID:PGID` with a read-only root filesystem, dropped
capabilities, and explicit resource limits. Narrow tmpfs mounts hold required
ephemeral writes. The one-shot API setup service initializes the LangGraph
checkpoint schema before both API workers, while Chainlit's `pre_start` hook
applies its independent UI migrations.

The API workers share PostgreSQL for both durable checkpoints and fail-fast
same-run coordination. Session-level advisory locks prevent two workers from
advancing the same interrupt run at once; a contended request fails instead of
waiting. No Redis service is required. The lock is held only while an API
request validates or advances the run, never while a human is deciding. A
per-process capacity gate also fails fast when its four lease slots are full,
leaving the fifth pool connection available for checkpoint I/O.

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
    five-connection pool and permits at most four simultaneous advisory leases,
    preserving one connection for checkpoint I/O. Psycopg recommends monitoring
    pool statistics and sizing from observed workload; see its
    [pool guidance](https://www.psycopg.org/psycopg3/docs/advanced/pool.html#pool-connection-and-sizing).

    The coordinator uses session-level advisory locks and must retain one
    database session for the whole lease. If a proxy such as PgBouncer sits in
    front of PostgreSQL, use session pooling or a direct coordinator connection;
    PgBouncer documents session advisory locks as unsupported in
    [transaction-pooling mode](https://www.pgbouncer.org/features.html#sql-feature-map-for-pooling-modes).

## What The Stack Demonstrates

- The API and Chainlit applications use their own lockfiles. The LGOS release
  workflow injects its tagged wheel into the API image, while development uses
  an editable parent checkout.
- Third-party services use pinned official images rather than being repackaged.
- Health checks, the one-shot API setup service, and Chainlit's `pre_start` job
  establish service and schema readiness.
- PostgreSQL provides both interrupt durability and cross-worker advisory
  coordination; the graph API needs no second persistence service.
- Read-only roots, dropped capabilities, tmpfs mounts, resource limits, and
  host-owned bind directories make operational assumptions visible.
- The API, UIs, and gateway communicate only through their documented network
  contracts.

!!! warning "Demo images are examples"

    The published images run the demo applications and graphs. They are not
    generic LGOS server images and should not be used as the base contract for
    an application that owns different graphs or dependencies.

Applications outside `demo/` own their container images and deployment model;
LGOS does not prescribe either. For exact demo commands and environment
variables, see [Demo Settings and Commands](reference.md).
