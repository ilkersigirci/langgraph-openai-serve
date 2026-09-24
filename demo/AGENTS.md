# Demo Agent Guidance

Each demo project is independent. Keep its dependencies and lockfile local to
that project. From the repository root, run
`just demo/check`; add `--editable` when validating the parent LGOS
checkout.

Use `../docs/demo/api.md` and `../docs/demo/graphs/index.md` for demo runs and
graph files, and `../docs/demo/reference.md` for demo settings and commands.
Read [Demo graph documentation](.agents/skills/demo_graph_doc.md) before
creating or substantially revising a page under `../docs/demo/graphs/`.
Check demo graph adapters before changing public graph APIs.

For live Chainlit or Open WebUI verification, use
[Demo browser checks](.agents/skills/browser-checks/SKILL.md). It owns the
Playwright setup, login flows, rendering checks, and common deployment
failures for both demo UIs.

For LiteLLM or Bifrost upgrades and workaround reviews, use
[Gateway upgrade](.agents/skills/gateway-upgrade/SKILL.md). Keep the upgrade
procedure in agent guidance; published docs describe the bundled gateway
behavior. LiteLLM's image pin belongs in `DEMO_LITELLM_IMAGE` there.

Compose files only pass environment variables through. Put defaults in
`demo/.env.example`; it is the source of truth.

Record each significant demo decision as a row in its component's section of
[Demo design choices](../docs/demo/design-choices.md).

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
  `gateway`, `files`, `responses`, `interrupts`, or `pipe`. Update `GENERIC_BUNDLE` and bundling tests
  together when adding a module.
- Validate OpenWebUI changes from `ui/openwebui/` with:
  `uv run --locked pytest`, `uv run --locked ruff check src tests`, and
  `uv run --locked ty check src`.

## Chainlit Utilities

- Read [Demo design choices](../docs/demo/design-choices.md) before changing the
  interrupt lifecycle, review element, or sibling HITL helper.
- Keep the Chainlit demo constrained to the compatible released `chainlit-utils`
  series from PyPI, with the exact release recorded in `uv.lock`; do not commit a
  local path source.
- Agents may change the sibling `../chainlit-utils` repository when reusable
  Chainlit behavior needs development. Test those unpublished changes in the
  demo with `uv run --with-editable "../../../../chainlit-utils[sso]" <command>` from
  `demo/ui/chainlit_ui/`.
- Keep using the editable overlay during joint development, then publish
  `chainlit-utils` and refresh the demo's PyPI constraint and lockfile when the
  changes are released.
- Keep generic helper and PostgreSQL token-storage tests in `chainlit-utils`;
  the demo owns LGOS settings, gateway wiring, and application-flow tests.
