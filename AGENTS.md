Guidance for coding agents in this repository. Keep this file operational;
use `docs/` for product explanations, examples, and detailed references.

## Start Here

- Read `README.md` for project purpose and the shortest example.
- Read `docs/index.md` only when you need the full docs map.
- Use `docs/tutorials/getting-started.md` for demo runs and demo graph files.
- Use `docs/reference.md` for API surface, settings, demo models, and commands.
- Use `docs/explanation/openai-compatibility.md` before changing API behavior;
  its citation ownership section defines the LGOS, Chainlit, and Open WebUI
  boundaries.

## Do

- Preserve OpenAI client compatibility as the only ingestion contract.
- Keep changes scoped to the affected package, demo, tests, or docs area.
- Add or update focused tests for behavior changes.
- For OpenAI route errors with known metadata, raise `OpenAIHTTPException` with
  `openai.types.shared.ErrorObject`.
- Check demo graph adapters before changing public graph APIs.

## Do Not

- Do not add project-specific chat envelopes, response shapes, headers, routes,
  or streaming events unless they remain reachable through `/v1`.
- Do not treat `curl` examples as a separate product contract; they are
  diagnostics only.
- Do not raise bare `HTTPException` from OpenAI route code when error metadata is
  known.
- Do not update dependencies, regenerate `uv.lock`, or touch `.env` unless the
  task requires it.

## Repo Map

- `src/langgraph_openai_serve/api/`: OpenAI-compatible routes and schemas.
- `src/langgraph_openai_serve/graph/`: graph registration, adapters, execution.
- `src/langgraph_openai_serve/openai_server.py`: FastAPI binding.
- `demo/`: runnable API and UI examples.
- `tests/`: pytest coverage.
