# Docker Compose

## Self-Contained Demo Layout

The demo uses three independent uv projects rather than a uv workspace:

| Project | Lockfile | Deployment |
| --- | --- | --- |
| `demo/api` | `demo/api/uv.lock` | `ghcr.io/ilkersigirci/lgos-demo-api` |
| `demo/ui/chainlit_ui` | `demo/ui/chainlit_ui/uv.lock` | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `demo/ui/openwebui` | `demo/ui/openwebui/uv.lock` | Local Function sync command |

Published project-owned images use only their project directories as build
contexts. The development override additionally supplies the parent LGOS
checkout as a named context for the API's editable install. The Open WebUI
integration uses the official Open WebUI image and keeps its Function sync
command local. There is no demo-wide `pyproject.toml`, uv workspace, shared
Python environment, or shared lockfile. Shared Compose-only configuration lives
under `demo/docker/`. The API package includes the compact Markdown corpus used
by `lgos-rag`.

## Compose Modes

!!! note "Docker Compose 5.3.0 or newer"

    The demo uses `pre_start` init containers to apply the API and Chainlit
    schema migrations before their services start.

Prepare the demo environment:

```bash
cd demo
cp .env.example .env
```

Set `PUID` and `PGID` in `.env` to the numeric host identity that owns the bind
directories. The checkout includes each empty service directory with a tracked
`.gitkeep`; service-created contents remain ignored.

=== "Published images"

    `compose.yaml` contains no local builds:

    ```bash
    docker compose -f compose.yaml up
    ```

    `DEMO_IMAGE_TAG` defaults to `latest`. Set it in `.env` to select one
    release tag for both project-owned demo images.

=== "Build demo projects"

    Apply the explicit development model from the LGOS repository checkout.
    The API and Chainlit services build locally from their Dockerfiles and
    lockfiles. The API image installs the parent LGOS checkout as an editable
    package:

    ```bash
    docker compose -f compose.yaml -f compose.dev.yaml up --build
    ```

    To watch for changes:

    ```bash
    docker compose -f compose.yaml -f compose.dev.yaml watch
    ```

    Changes to either the demo API source or the parent LGOS package restart
    the API against their narrow, read-only bind mounts. Both packages are
    installed editable in the development image. Dependency metadata and
    lockfile changes rebuild the image.

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

=== "API"

    ```bash
    docker compose -f compose.yaml up -d lgos-demo-api
    ```

    OpenAI base URL: `http://localhost:8000/v1`

=== "Bifrost"

    ```bash
    docker compose -f compose.yaml up --wait bifrost
    ```

    - Standard inference, without client events:
      `http://localhost:8081/v1` with `openai/`-prefixed models
    - Detailed discovery or event-enabled inference:
      `http://localhost:8081/openai_passthrough/v1` with unprefixed models

    From the package repository, run `make test-bifrost` to verify detailed
    model metadata through the proxy. See
    [Bifrost Gateway](bifrost.md).

=== "Chainlit"

    ```bash
    docker compose -f compose.yaml up -d lgos-chainlit
    ```

    Chainlit: `http://localhost:5000`

    Configure its signing secret as described in the
    [Chainlit client](chainlit.md).

=== "Open WebUI"

    ```bash
    docker compose -f compose.yaml up --wait open-webui
    make sync-openwebui
    ```

    Open WebUI: `http://localhost:8080`

    Compose runs the official Open WebUI image. The local sync command creates
    or updates the bundled Functions. See the
    [Open WebUI Functions](open-webui.md).

PostgreSQL checkpoints, Bifrost state, and Open WebUI state use host bind mounts
under `demo/docker/volumes/`; the Compose model declares no named volumes. Every
service runs as `PUID:PGID` with a read-only root filesystem, dropped
capabilities, and explicit resource limits. Narrow tmpfs mounts hold required
ephemeral writes. The API and Chainlit `pre_start` hooks apply their independent
schema migrations.

## What The Stack Demonstrates

- The API and Chainlit applications use their own lockfiles. The production API
  image uses the locked PyPI release in a minimal runtime, while the development
  image installs both the API and the parent LGOS checkout as editable packages.
- Third-party services use pinned official images rather than being repackaged.
- Health checks and `pre_start` jobs establish service and schema readiness.
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
