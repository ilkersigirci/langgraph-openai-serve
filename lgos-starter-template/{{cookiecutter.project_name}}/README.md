# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Start

Install Python {{ cookiecutter.python_version }}, [uv](https://docs.astral.sh/uv/),
Bash, and [Just 1.58.0+](https://just.systems/). Then:

```bash
just setup
```

This installs dependencies, creates `uv.lock`, and copies `.env.example` to
`.env` if it does not exist. Commit `uv.lock` with your application. Edit
`APP_OPENAI_API_KEY`, `APP_OPENAI_BASE_URL`, and `APP_OPENAI_MODEL` in `.env`.

```bash
just run
```

`just run` passes Uvicorn options through: `just run --reload` restarts the
server when code changes, which also discards in-process interrupt state.

Call the graph from another terminal:

```bash
uv run --locked --env-file .env examples/responses.py
uv run --locked --env-file .env examples/responses.py --stream
```

The OpenAI base URL is `http://localhost:8000/v1`; `simple-graph` and `approval`
are the model IDs. `examples/chat.py` demonstrates Chat Completions with the
same `--stream` option, and `examples/approval.py` pauses for a decision.

`just run` starts `lgos serve`, which provides JSON logging, OpenTelemetry,
Langfuse, PostgreSQL interrupts, and Hatchet background execution. Development
needs no services; `.env` selects PostgreSQL, Hatchet, and Langfuse when you
configure them.

## Develop

```bash
just check
just docs --serve
```

Default tests use a fake model and the server's in-process resources. They
need no provider credentials or running services. Read [AGENTS.md](AGENTS.md),
[code style](.agents/CODE_STYLE.md), and [test guidance](tests/README.md)
before changing the project.

Use [the graph guide](docs/graphs.md) to add your own graphs, and
[deployment](docs/deployment.md) for containers, PostgreSQL, Hatchet, and
observability. This application has no built-in authentication. Put a gateway
with bearer-token authentication in front of it before exposing it to clients
outside a trusted environment.

Generated from the LGOS starter template. This project is independent:
changing the source template does not overwrite your application. Review LGOS
release notes before raising its `pyproject.toml` constraint, then run
`just check`.
