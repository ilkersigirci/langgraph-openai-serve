# LGOS Starter Template

Generate an independent LangGraph application served by `lgos serve`, with
typed graph settings, tests, developer guidance, Docker, and Zensical
documentation.

```bash
uvx cookiecutter https://github.com/ilkersigirci/langgraph-openai-serve.git \
  --directory lgos-starter-template
```

Add `--checkout <tag>` to generate the template of a specific release. The
[starter guide](../docs/starter-template.md) describes the options and the
first run.

## Maintain the template

From this repository:

```bash
just lgos-starter-template/check
```

The template-maintenance environment has its own `pyproject.toml` and lockfile.
Tests render the template into temporary directories, validate generated
Python/configuration, and check formatting. CI additionally installs generated
applications outside this checkout with this checkout's LGOS, runs their tests,
types, docs, and Compose checks, and builds their wheels. Their image is built
against the LGOS release that the template's constraint selects and must start
healthy beside PostgreSQL.

Hosting lives in LGOS (`langgraph_openai_serve.server`); generated projects own
only their graphs, settings, tests, and docs, adapted from `demo/` conventions.
Do not generate this template dynamically from the showcase graph catalog.
