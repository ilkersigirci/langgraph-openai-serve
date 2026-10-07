# Generate A Starter Application

The Cookiecutter template creates an independent project that
[`lgos serve`](how-to-guides/server.md) runs: a simple streaming graph, an
approval graph that pauses for a human, typed graph settings, pytest coverage,
Ruff, type checking, agent guidance, Docker, and documentation. The server
supplies logging, OpenTelemetry, Langfuse, PostgreSQL interrupts, and Hatchet
background execution, so the project holds only its graphs.

## Generate

Install [uv](https://docs.astral.sh/uv/), Bash, and
[Just 1.58.0 or newer](https://just.systems/).

=== "From GitHub"

    ```bash
    uvx cookiecutter https://github.com/ilkersigirci/langgraph-openai-serve.git \
      --directory lgos-starter-template
    ```

=== "From this checkout"

    ```bash
    uvx cookiecutter ./lgos-starter-template
    ```

??? note "Generate the template of a specific release"

    The GitHub command uses the template on `main`. To use the template of an
    earlier LGOS release instead, add its
    [release tag](https://github.com/ilkersigirci/langgraph-openai-serve/tags):

    ```bash
    uvx cookiecutter https://github.com/ilkersigirci/langgraph-openai-serve.git \
      --directory lgos-starter-template --checkout <tag>
    ```

Choose the project name, Python package name, description, author, email,
Python version, license, and CI provider: GitHub Actions or GitLab CI. Either
pipeline runs the lint, test, and documentation checks as parallel jobs. To
generate non-interactively:

```bash
uvx cookiecutter ./lgos-starter-template --no-input project_name=research-api
```

## Run

```bash
cd research-api
just setup
just run
```

Setup installs dependencies, creates `uv.lock`, and copies `.env.example` to
`.env` if it is absent. Commit the lockfile. Set the upstream model endpoint,
key, and model in `APP_OPENAI_*`. Development needs no other service: interrupt
state and background runs stay in the server process.

Call the API from another terminal:

```bash
uv run --locked --env-file .env examples/responses.py
uv run --locked --env-file .env examples/responses.py --stream
uv run --locked --env-file .env examples/approval.py
```

## Develop And Deploy

Run `just check` for the generated tests, formatting, lint, types, and strict
documentation build. Tests use a deterministic model and the server's
in-process resources, so they need no live services.

Read `AGENTS.md`, `.agents/CODE_STYLE.md`, and `tests/README.md` for development
guidance. Register your own graphs in `registry.py`; the API and the Hatchet
worker both run that factory.

`just up` builds the image and starts it with a PostgreSQL service for durable
interrupts. Set `LGOS_BACKGROUND=hatchet` with the credentials of an existing
Hatchet tenant to run background Responses in a worker, and
`LGOS_ENABLE_LANGFUSE=True` with an existing Langfuse project to trace graph
runs. Authentication belongs to a gateway in front of the server, as in the
[package authentication guide](how-to-guides/authentication.md).

Generated projects do not depend on this repository's files. They own their
source and lockfile; later template changes do not overwrite them.
