# LGOS demos

This directory is a self-contained demo distribution for
`langgraph-openai-serve`. It deliberately is not a uv workspace: each
application or integration tool has its own `pyproject.toml`, `.venv`, and
`uv.lock`. Locked installs, tests, published Docker builds, and the published
Compose stack need no files outside this directory.

The API resolves `langgraph-openai-serve` from PyPI and packages the default
`lgos-rag` Markdown corpus inside `lgos_demo_api`. The development Compose
override builds a local API image and installs both the API and parent LGOS
checkout as editable packages without changing the locked production
dependency source.

| Project | Purpose | Deployment |
| --- | --- | --- |
| `api` | Example LangGraph API | `ghcr.io/ilkersigirci/lgos-demo-api` |
| `ui/chainlit_ui` | Chainlit client | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `ui/openwebui` | Open WebUI Function sync | Local uv command |

Shared Compose-only assets live under `docker/`; the Bifrost gateway
configuration is at `docker/bifrost/config.json`. Compose runs two LGOS API
containers behind its `lgos-a/` and `lgos-b/` model prefixes. Both use the demo
image today; either service can be replaced by an independently locked
application image when graph dependencies conflict.

Compose persists PostgreSQL, Bifrost, and Open WebUI state as ignored host bind
mounts under `docker/volumes/`. Each service directory is tracked with a
`.gitkeep`; runtime contents remain ignored. Services run as the configured
`PUID:PGID` with read-only container filesystems, limited writable tmpfs paths,
dropped Linux capabilities, and explicit CPU, memory, PID, and file-descriptor
limits.

## Run containers independently

Copy the shared environment template and configure any required credentials:

```bash
cp .env.example .env
```

Run the published `lgos-a` container on port 3004:

```bash
make run-api
```

Run the same published image as `lgos-b` on port 3005:

```bash
make run-api-b
```

Run Chainlit and its Compose dependencies on port 3002:

```bash
make run-chainlit
```

With Open WebUI running, synchronize the bundled Functions:

```bash
make sync-openwebui
```

Compose starts each selected service's dependencies and applies the API and
Chainlit migrations through their `pre_start` hooks.

## Run local processes

Start PostgreSQL for the local API and UI processes:

```bash
docker compose -f compose.yaml up -d lgos-db
```

The local targets use the independently locked projects. The API additionally
overlays the parent LGOS checkout as an editable dependency:

```bash
make run-api-local
make run-api-b-local
make run-chainlit-local
```

Run each long-lived process in a separate terminal.

## Run the stack

Use the two published demo images and the official third-party images:

```bash
make compose
```

The stack publishes Bifrost on port 3000, PostgreSQL on 3001, Chainlit on
3002, Open WebUI on 3003, `lgos-a` on 3004, and `lgos-b` on 3005.

From the LGOS source checkout, build the two project-owned application images
from their own lockfiles and run the API against the editable parent package:

```bash
make compose-dev
```

Set `PUID` and `PGID` in `.env` to the host identity that owns
`docker/volumes/`; the example values are `1000:1000`.

## Automation

When this directory is copied to a repository root, its `.github/workflows`
files test all three locked projects and build the API and Chainlit images. The
LGOS source repository carries thin root workflow wrappers while the directory
is kept in-tree. Both sets of workflows use the composite actions owned by this
directory; the root test wrapper checks a copy outside the package checkout and
also runs the API against the current LGOS source.

Pull requests validate changed image contexts without publishing. Pushes and
tag creation do not run the image publishing jobs. Creating a GitHub release
such as `v0.1.0` publishes `latest`, `0.1.0`, and the immutable commit tag for
both images.

Published images include an SBOM and maximum BuildKit provenance. Actions are
pinned to full commit hashes, credentials are not persisted after checkout, and
GHCR write permission is granted only to publishing jobs.

Run every locked test, lint, formatting, and Compose check with:

```bash
make check
```

The directory is licensed under the included [MIT License](LICENSE).
