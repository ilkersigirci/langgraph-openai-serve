# Starter Template Maintenance

- Read the root `AGENTS.md`, `.agents/CODE_STYLE.md`, and `tests/README.md`.
- `{{cookiecutter.project_name}}/` and `hooks/` contain unrendered Jinja.
  Test and lint generated projects, not the raw Python templates.
- Keep local imports parenthesized with trailing commas so short and long
  Python package names both render with valid formatting.
- Run `just lgos-starter-template/check` from the repository root.
- Generated projects contain graphs, their settings, tests, and docs. Hosting
  belongs to `lgos serve` in the package; do not copy it back into the template.
- CI validates generated projects outside this checkout against this checkout's
  LGOS, so template and server changes land together. The image job uses the
  released LGOS, so it passes again once a needed server change is published.
- Langfuse and Hatchet are external services. Include their client configuration
  and the application worker, not server deployments.
- Template-maintenance dependencies live here; generated applications own their
  dependencies and create their lockfile with `just setup`.
