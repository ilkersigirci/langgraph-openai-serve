# Demo Settings And Commands

This reference describes the independently locked projects and Compose stack
under `demo/`. These commands and `DEMO_*` settings are not part of the
`langgraph-openai-serve` package API.

## Projects

| Path | Purpose | Imports LGOS? |
| --- | --- | --- |
| `demo/api` | Example FastAPI and LangGraph application | Yes, from PyPI by default |
| `demo/ui/chainlit_ui` | Persistent OpenAI-protocol client | No |
| `demo/ui/openwebui` | Open WebUI Function sources and sync command | No |
| `demo/docker` | Compose-only Bifrost configuration and service data directories | No |

Each Python project has its own `pyproject.toml`, virtual environment, and
`uv.lock`; `demo/` deliberately is not a uv workspace.

## Common Commands

Run these from `demo/` after copying `.env.example` to `.env`:

| Command | Purpose |
| --- | --- |
| `make run-api` | Set up PostgreSQL checkpoints and run the API locally |
| `make run-chainlit` | Apply Chainlit migrations and run the UI locally |
| `make sync-openwebui` | Create or update the bundled Open WebUI Functions |
| `make compose` | Run the stack with published project-owned images |
| `make compose-dev` | Build local images; run the API and LGOS packages editable |
| `make sync` | Synchronize all three projects from their lockfiles |
| `make test` | Test all three projects from their lockfiles |
| `make lint` | Check all three projects with Ruff |
| `make check` | Run tests, lint, formatting checks, and Compose validation |

## Stack Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_IMAGE_TAG` | `latest` | Tag selected for both project-owned demo images |
| `PUID` | `1000` | Host user ID used by Compose services |
| `PGID` | `1000` | Host group ID used by Compose services |

## Demo API Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_API_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible base URL |
| `DEMO_API_OPENAI_API_KEY` | `DUMMY` | Upstream key for provider-backed graphs |
| `DEMO_API_OPENAI_MODEL` | `gpt-5.4-mini` | Upstream generation model |
| `DEMO_API_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used by `lgos-rag` |
| `DEMO_API_POSTGRES_URI` | `postgresql://lgos:lgos@localhost:5432/lgos` | Checkpoint database |

The API also reads the package-owned `LGOS_OPENAI_API_PREFIX` and
`LGOS_OPENAI_API_DOCS_ENABLED` settings documented in the package
[Reference](../reference.md#settings).

See [Chainlit settings](chainlit.md#settings-reference),
[Open WebUI setup](open-webui.md#setup), and the [example graph catalog](graphs.md)
for component-specific details.
