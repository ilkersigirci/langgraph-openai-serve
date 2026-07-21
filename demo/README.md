# LGOS demos

This directory is a self-contained demo distribution for
`langgraph-openai-serve`. It deliberately is not a uv workspace: each
application or integration tool has its own `pyproject.toml`, `.venv`, and
`uv.lock`. Locked installs, tests, Docker builds, and Compose need no files
outside this directory.

The API resolves `langgraph-openai-serve` from PyPI and packages the default
`lgos-rag` Markdown corpus inside `lgos_demo_api`. A temporary editable overlay
can test an LGOS source checkout, but it is never part of the demo metadata,
lockfile, image, or default runtime.

| Project | Purpose | Deployment |
| --- | --- | --- |
| `api` | Example LangGraph API | `ghcr.io/ilkersigirci/lgos-demo-api` |
| `ui/chainlit_ui` | Chainlit client | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `ui/openwebui` | Open WebUI Function sync | Local uv command |

Shared Compose-only assets live under `docker/`; the Bifrost gateway
configuration is at `docker/bifrost/config.json`.

Compose persists PostgreSQL, Bifrost, and Open WebUI state as ignored host bind
mounts under `docker/volumes/`. Each service directory is tracked with a
`.gitkeep`; runtime contents remain ignored. Services run as the configured
`PUID:PGID` with read-only container filesystems, limited writable tmpfs paths,
dropped Linux capabilities, and explicit CPU, memory, PID, and file-descriptor
limits.

## Run independently

Copy the shared environment template and configure any required credentials:

```bash
cp .env.example .env
```

Run the API:

```bash
make run-api
```

Run Chainlit in another terminal:

```bash
make run-chainlit
```

With Open WebUI running, synchronize the bundled Functions:

```bash
make sync-openwebui
```

## Run the stack

Use the two published demo images and the official third-party images:

```bash
docker compose -f compose.yaml up
```

Build the two project-owned application images from their own lockfiles and
Docker contexts:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
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

Pull requests validate changed image contexts without publishing. Pushes to
`main` publish `latest` and an immutable `sha-<commit>` tag. Tags such as
`v0.1.0` publish `0.1.0` and the immutable commit tag for both images.

Published images include an SBOM and maximum BuildKit provenance. Actions are
pinned to full commit hashes, credentials are not persisted after checkout, and
GHCR write permission is granted only to publishing jobs.

Run every locked test, lint, formatting, and Compose check with:

```bash
make check
```

The directory is licensed under the included [MIT License](LICENSE).
