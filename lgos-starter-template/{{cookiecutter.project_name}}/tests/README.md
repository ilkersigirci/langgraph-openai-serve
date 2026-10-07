# Test Guide

Run `just test` for the service-free suite and `just check` for all local
quality checks. Tests assert observable behavior through the OpenAI SDK and
the real server app, using deterministic model responses. Run a larger suite in
parallel with `just test -n auto`, so keep each test independent of the others.

- Tests read `.env` like the application, so run `just setup` first. Pytest
  replaces `APP_OPENAI_API_KEY` with a placeholder, so tests never see a real key.
- `tests/support.py` builds the app with a fake model and explicit server
  settings, so `LGOS_*` values from `.env` never reach a test. Its in-process
  resources need no database, Hatchet tenant, or Langfuse project.
- AnyIO automatically discovers async tests. Keep the function-scoped
  `anyio_backend` fixture set to `asyncio`; do not add per-test AnyIO markers.
- Keep resources and their cleanup together in async contexts or yield fixtures.
- Put fixtures in the nearest `conftest.py`; do not import that module.
- Inject dependencies through graph factories and `create_registry`. Use
  callbacks to inspect model inputs when that input is the behavior being tested.
- Use bounded timeouts and events for concurrency tests. Avoid timing sleeps
  used solely to make tasks interleave.

## Restricted coding-agent sandboxes

If async tests stall, inspect a run with `pytest -o faulthandler_timeout=5`.
Some sandboxes block asyncio's internal socketpair wakeups, leaving an event
loop waiting while a worker thread has finished. Stop the stalled command and
use the coding tool's supported unsandboxed execution mechanism, following its
permission policy. Explain the environment restriction if approval is needed.
Do not add sleeps, heartbeat timers, or application changes to mask it.
