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
| `files_api` | OpenAI Files API backed by S3 | `ghcr.io/ilkersigirci/lgos-files-api` |
| `ui/chainlit_ui` | Chainlit client | `ghcr.io/ilkersigirci/lgos-chainlit` |
| `ui/openwebui` | Open WebUI Function sync and raw-upload policy | Local uv command and official-image bind mount |

Compose service fragments live under `docker/apps/`, with entrypoints and
overlays under `docker/compose/`. Bifrost and LiteLLM gateway configurations
live under `docker/configs/`. Compose runs the demo API image as two
independently addressable services, `lgos-a` and `lgos-b`. They
serve the same graphs under separate provider identities so the stack can
exercise native Responses routing through either gateway. The independent
`lgos-files-api` image provides the shared S3-backed Files service. LiteLLM
1.99.1 and Bifrost v2.0.0 are both first-class UI gateways. Set
`OPENAI_GATEWAY_TYPE=litellm|bifrost` once for Chainlit and Open WebUI. Neither
UI connects to an upstream container directly. Responses and Files use each
gateway's normal OpenAI routes; a catalog-only client uses pass-through model
detail where needed to preserve LGOS descriptions, features, and settings.

Current verification exposes narrower upstream normalization limitations.
Bifrost v2.0.0's normalized `/openai/v1` route preserves the tested native
Responses fields, file input, commentary `phase`, and continuation, but not
LGOS model-detail extensions or upstream error metadata. LiteLLM 1.99.1 still
synthesizes wildcard Responses streams and rewrites standard error metadata.
Bifrost's raw pass-through and LiteLLM's authenticated pass-through both
preserve the full tested contract for protocol diagnostics. UI inference does
not use either pass-through: it exercises LiteLLM's managed Responses route or
Bifrost's native Responses route according to `OPENAI_GATEWAY_TYPE`. Run
`make test-bifrost` and `make test-litellm` from the repository root for the
current compatibility matrix.

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

Run the central Files API on port 3006:

```bash
make run-files
```

Run either native Responses gateway with its graph API dependencies:

```bash
make run-bifrost
make run-litellm
```

Run Chainlit and its Compose dependencies on port 3002:

```bash
make run-chainlit
```

With Open WebUI running, synchronize the Functions and generated Workspace
Models:

```bash
make sync-openwebui
```

Compose starts each selected service's dependencies. One API setup job
initializes the LangGraph checkpointer and Store schemas; Chainlit applies its
own migrations through `pre_start`.

## Run local processes

Start PostgreSQL for the local API and UI processes:

```bash
docker compose -f docker/compose/demo.yml up -d lgos-db
```

The local targets use the independently locked projects. The API additionally
overlays the parent LGOS checkout as an editable dependency:

```bash
make run-api-local
make run-api-b-local
make run-files-local
make run-chainlit-local
```

Run each long-lived process in a separate terminal.

## Run the stack

Use the published demo images and the official third-party images:

```bash
make compose
```

The stack publishes Bifrost on port 3000, PostgreSQL on 3001, Chainlit on
3002, Open WebUI on 3003, `lgos-a` on 3004, `lgos-b` on 3005, the Files API on
3006, and LiteLLM on 3007. The selected UI gateway is controlled by
`OPENAI_GATEWAY_TYPE`.

From the LGOS source checkout, build the project-owned application images
from their own lockfiles and run the API against the editable parent package:

```bash
make compose-dev
```

Run the published stack with the optional local OpenTelemetry Collector:

```bash
make compose-otel
```

For local source changes with the same overlay, use `make compose-otel-dev`.
See the repository's [demo OpenTelemetry guide](../docs/demo/opentelemetry.md)
for signal ownership and the external gateway contract.

Set `PUID` and `PGID` in `.env` to the host identity that owns
`docker/volumes/`; the example values are `1000:1000`.

## Automation

When this directory is copied to a repository root, its `.github/workflows`
files test all four locked projects and build the API, Files API, and Chainlit
images. The
LGOS source repository carries thin root workflow wrappers while the directory
is kept in-tree. Both sets of workflows use the composite actions owned by this
directory; the root test wrapper checks a copy outside the package checkout and
also runs the API against the current LGOS source.

Each workflow layout lets `setup-uv` discover the version requirement at its
repository root: `pyproject.toml` for LGOS and `uv.toml` for a copied demo.

Pull requests validate changed image contexts without publishing. Creating a
GitHub release such as `v0.8.0` publishes `0.8.0` and `latest` for all three
images.
In the LGOS repository, the API image includes a wheel built from the tagged
checkout instead of waiting for PyPI.

Published images include an SBOM and maximum BuildKit provenance. Actions are
pinned to full commit hashes, credentials are not persisted after checkout, and
GHCR write permission is granted only to publishing jobs.

Run every locked test, lint, formatting, and Compose check with:

```bash
make check
```

The directory is licensed under the included [MIT License](LICENSE).
