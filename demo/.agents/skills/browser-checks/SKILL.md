---
name: browser-checks
description: >-
  Run focused live browser checks against the demo Chainlit or Open WebUI
  clients when a task requires browser-visible verification or UI diagnosis.
---

# Demo Browser Checks

Use the installed global `playwright-cli` skill for CLI syntax and capabilities;
this skill owns the repository-specific workflow. Run commands from the
repository root, with paths below interpreted from there.

Use browser checks only for behavior that depends on a browser, such as login,
storage, streaming UI, downloads, rendering, or interaction. Prefer focused
tests, `curl`, or service APIs for protocol and payload checks. Do not add
Playwright to a demo project or change its lockfile for an agent-run check.

## Prepare

- Inspect the running stack and ports with `docker compose ls` and `docker ps`.
  The default URLs are Chainlit `http://localhost:3002` and Open WebUI
  `http://localhost:3003`; honor deployment overrides.
- After an Open WebUI Function change, run `just demo/sync-openwebui`. After a
  Chainlit source change, rebuild and recreate only `lgos-chainlit` with the
  running stack's Compose files and development overlay; its source is baked
  into the image even during development.
- Open isolated, headless Chromium with the checked-in config. Use a unique,
  descriptive session for each UI and keep that session throughout its flow:

```bash
playwright-cli -s=lgos-openwebui open http://localhost:3003 \
  --config=demo/.agents/skills/browser-checks/cli.config.json
```

  Replace the example name with a task-specific session name. Close that exact
  session when finished; never use `close-all`, `kill-all`, or another agent's
  session on the shared host.
- If `open` reports a missing browser or Linux library, provision the host once
  with `playwright-cli install-browser chromium --with-deps`. Do not reinstall
  it during routine checks.
- If a restricted sandbox cannot write the CLI daemon cache, set
  `PWTEST_DAEMON_SESSION_DIR` to a unique task directory under `/tmp`; do not
  override `HOME` or a shared cache directory.

## Keep Checks Focused and Safe

- Start with `find` for expected text. Take a targeted or depth-limited
  `snapshot` only when element references or nearby structure are needed. Use
  `--raw` when only a compact evaluation result is useful.
- Take screenshots only for visual assertions such as layout, clipping, or
  chart output. Use tracing or video only to investigate a failure. Keep
  screenshots, snapshots, traces, downloads, and temporary auth state under
  `/tmp`.
- Test chats remain in user history. Start a new chat with a harmless,
  read-only prompt such as “Show the quarterly revenue chart,” and report its
  URL.
- Use an ephemeral browser profile. Do not attach to a personal browser or
  reuse personal cookies. Keep credentials, tokens, auth state, secret-bearing
  responses, and sensitive chat content out of commands, logs, screenshots,
  and Git.

## Open WebUI

Authenticate with `DEMO_OPENWEBUI_ADMIN_EMAIL` and
`DEMO_OPENWEBUI_ADMIN_PASSWORD` without placing their values in CLI arguments
or output. For API-assisted `/api/v1/auths/signin`, use a temporary external
helper that reads the demo environment and writes a mode-`0600` storage-state
file under a unique `/tmp` path. Load it with `state-load` and delete it after
closing the session. The `run-code` evaluator cannot read the CLI process
environment.

Open `/?model=lgos.lgos-a/persistent-plot-agent`. For another graph, use the
Workspace Model ID from `/api/models`. Fill `#chat-input`, submit the prompt,
and wait for a completed assistant-role reply. Plotly renders inside the
message iframe, so inspect the matching frame and its height.

## Chainlit

With the default mock login, fill `input[name=email]` and
`input[name=password]` with any nonempty demo values, then select `Sign In`.
This uses the shared `demo-user`; OAuth deployments require their configured
login flow.

Select the current profile label, then choose
`lgos-a/persistent-plot-agent` by exact text. Wait for the profile settings
reload and `#chat-input` readiness before submitting because switching profiles
resets the session. Native Plotly elements render in the main page.

## Verify and Diagnose

- Wait for `.js-plotly-plot` and the completed assistant reply. Use a targeted
  evaluation to confirm chart data, exercise one interaction such as hover or
  zoom, and take one screenshot for the visual assertion. In Open WebUI, also
  verify the iframe height so the chart is not clipped.
- Reload once only when persistence is part of the requested check. Do not
  infer completion by counting prompt text: Open WebUI can repeat it in the
  sidebar title. Assert an assistant-role message in the chat region or the
  saved chat's `history.messages`, then verify it after reload.
- On failure, inspect the visible error first, then the relevant console entry,
  numbered request, and affected service logs. Avoid broad dumps.
- Chainlit `provider query parameter is required` means generated-file
  downloads must use `files_request()` and its provider, as uploads already do.
- Open WebUI `Model not found` requires checking that the `generic` Function is
  active and its Workspace Model's base model exists. A failed Function import
  can disable it; a successful sync preserves that state, so re-enable it after
  repairing the import.
- A clipped Open WebUI chart requires the native `iframe:height` notification
  used by Open WebUI's `FullHeightIframe` component.

Stop after the requested behavior and any requested persistence check pass.
Close the named session, remove sensitive temporary state, and report the URL,
assertions, result, and relevant `/tmp` artifact paths. If a live check is not
available, report that explicitly instead of treating source inspection as a
passing check.
