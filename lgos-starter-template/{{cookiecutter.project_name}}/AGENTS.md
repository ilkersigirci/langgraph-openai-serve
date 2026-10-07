# Coding Agent Guidance

- Read `README.md`, `.agents/CODE_STYLE.md`, and `tests/README.md` first.
- Keep this file operational. Put product explanations in `docs/`.
- Run `just check` before finishing code changes. For documentation, preview
  with `just docs --serve`, then run `just docs`.
- Preserve the OpenAI client contract under `/v1`. `lgos serve` owns hosting:
  logging, telemetry, persistence, and background execution. Keep graph logic
  under `graphs/` and register graphs in
  `src/{{ cookiecutter.project_slug }}/registry.py`; the API and Hatchet worker
  both run that factory.
- Compile interrupt graphs with `resources.checkpointer` and pass
  `resources.run_coordinator` to the registry. Keep interrupted nodes free of
  non-idempotent side effects.
- Raise LGOS `InvalidRequestError` for request validation and `GraphError` for
  graph configuration/output errors.
- Keep graph settings under `APP_`; LGOS owns the `LGOS_`, `UVICORN_`,
  `LANGFUSE_`, `HATCHET_`, and `OTEL_` namespaces.
- `.env.example` documents deployment defaults; `.env` contains local secrets.
  Do not read, modify, or commit secrets unless the task requires it.
- Use `uv sync --locked` after initial setup. Change dependencies and the
  lockfile together only when needed for the task.
- Add focused tests for observable behavior. Inject models through
  `create_registry`; default tests must not call external services.
- Native Zensical tabs, admonitions, and details are preferred over custom HTML.
