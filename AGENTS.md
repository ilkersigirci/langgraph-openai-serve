Guidance for coding agents in this repository. Keep this file operational;
use `docs/` for product explanations, examples, and detailed references.
Extended coding-agent guidance belongs under `.agents/`; use this file as its
entry point.

## Start Here

- Read `README.md` for project purpose and the shortest example.
- Read `.agents/CODE_STYLE.md` before writing or modifying code.
- Read `tests/README.md` before running or changing any pytest suite. Its
  guidance applies to `tests/`, `demo/api/tests/`,
  `demo/ui/chainlit_ui/tests/`, and `demo/ui/openwebui/tests/`, including the
  async-test guidance for restricted coding-agent sandboxes.
- Read `demo/AGENTS.md` before changing files under `demo/`; it documents the
  self-contained demo projects and the modular OpenWebUI bundling contract.
- Read `docs/index.md` only when you need the full docs map.
- Use `docs/getting-started.md` for the minimal package application.
- Use `docs/demo/api.md` and `docs/demo/graphs/index.md` for demo runs and graph
  files.
- Read `demo/.agents/skills/demo_graph_doc.md` before creating or substantially
  revising a page under `docs/demo/graphs/`.
- Use `docs/reference.md` for the package API and settings; use
  `docs/demo/reference.md` for demo settings and commands.
- Use `docs/explanation/openai-compatibility.md` before changing API behavior;
  its citation ownership section defines the LGOS, Chainlit, and Open WebUI
  boundaries.

## Do

- Preserve OpenAI client compatibility as the only ingestion contract.
- Keep changes scoped to the affected package, demo, tests, or docs area.
- Add or update focused behavior tests; do not add tests that merely mirror
  implementation details or detect behavior-preserving refactors. See
  `.agents/CODE_STYLE.md` for test-design guidance.
- For OpenAI route errors with known metadata, raise `OpenAIHTTPException` with
  `openai.types.shared.ErrorObject`.
- Check demo graph adapters before changing public graph APIs.


## Documentation Style

- Prefer native Zensical Markdown features enabled in `zensical.toml`; avoid
  custom HTML, CSS, or JavaScript when a native component fits.
- Use content tabs (`===`) for equivalent languages, clients, or modes;
  admonitions (`!!!`) for important notes; and collapsible details (`???`) for
  optional configuration or diagnostics.
- Use Mermaid for architecture and multi-step flows, keep diagrams compact, and
  verify them in the browser because the strict build does not parse Mermaid.
- Add language identifiers to code fences and use code annotations or tooltips
  only when they clarify details that do not belong in the main flow.
- Preview documentation with `make doc-serve`, then run `make doc-build` before
  finishing a docs change.

## Do Not

- Do not add project-specific chat envelopes, response shapes, headers, routes,
  or streaming events unless they remain reachable through `/v1`.
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
