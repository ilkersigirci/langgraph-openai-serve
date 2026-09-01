# Demo Agent Guidance

Each demo project is independent. Keep its dependencies and lockfile local to
that project; run demo-wide checks with `make -C demo test` and
`make -C demo lint type-check`.

## Modular OpenWebUI Function

- The source of truth for the Generic Function is
  `ui/openwebui/src/lgos_openwebui/functions/generic/`. Do not recreate or
  maintain a generated `generic.py` file.
- `generic/function.py` contains the OpenWebUI frontmatter. Its first line must
  be exactly `"""`; code after the frontmatter is for normal package imports
  only and is not included in the deployed source.
- `bundle.py` concatenates modules in `GENERIC_BUNDLE` order, removes only
  relative imports, adds source markers, and compiles the result. Keep imports
  acyclic and define names before they are used in that order. Third-party
  imports remain in the bundle.
- The bundle is one Python namespace. Do not define duplicate top-level names
  across modules; later definitions or imports can overwrite earlier ones.
  Avoid `from __future__` imports, dynamic/local package imports, `__file__`,
  and other module-boundary assumptions in bundled modules.
- Keep behavior in its responsibility module: `contracts`, `api`, `metadata`,
  `events`, `interrupts`, or `pipe`. Update `GENERIC_BUNDLE` and bundling tests
  together when adding a module.
- Validate OpenWebUI changes from `ui/openwebui/` with:
  `uv run --locked pytest`, `uv run --locked ruff check src tests`, and
  `uv run --locked ty check src`.

## Chainlit Utilities

- Keep the Chainlit demo pinned to the released `chainlit-utils` package from
  PyPI; do not commit a local path source.
- Agents may change the sibling `../chainlit-utils` repository when reusable
  Chainlit behavior needs development. Test those unpublished changes in the
  demo with `uv run --with-editable ../../../../chainlit-utils <command>` from
  `demo/ui/chainlit_ui/`.
- Keep using the editable overlay during joint development, then publish
  `chainlit-utils` and update the demo's PyPI pin when the changes are released.
