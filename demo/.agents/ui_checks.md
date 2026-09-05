# Browser UI checks

Use a browser to verify UI changes through the running demo and its configured
gateway. An API response alone does not verify file downloads or rendering.
Use an available browser tool, or a temporary Playwright script under `/tmp`.
Test chats are saved in the user's history: identify them in your report and
prefer a new chat with a read-only prompt such as “Show the quarterly revenue
chart.” Keep credentials, tokens, and browser auth state out of logs and Git.

## Prepare

- Check the running stack and ports with `docker compose ls` and `docker ps`.
  Default UI URLs are Chainlit `http://localhost:3002` and Open WebUI
  `http://localhost:3003`; use the actual deployment settings if overridden.
- After Open WebUI Function changes, run `make -C demo sync-openwebui` from the
  repository root. After Chainlit source changes, rebuild and recreate only
  `lgos-chainlit`, using the running stack's Compose files and development
  overlay. Chainlit source is baked into its image, even in development.
- For Playwright, run these commands from the repository root. The overlay
  makes the Open WebUI settings loader available without changing dependencies:

```bash
uv run --no-project --with playwright playwright install chromium
uv run --directory demo/ui/openwebui --locked --with playwright \
  --env-file ../../.env python /tmp/ui_check.py
```

Use `sync_playwright()`, `p.chromium.launch(headless=True)`, and
`browser.new_page(viewport={"width": 1400, "height": 1000})`. Close the browser
when finished. If Chromium reports missing Linux libraries, use Playwright's
`install-deps chromium` where permitted; avoid hard-coded browser cache paths.
See [Playwright browser setup](https://playwright.dev/python/docs/browsers).

## Open WebUI

Authenticate with the configured demo account through the native sign-in API,
then open a new chat. This snippet assumes a Playwright `page`:

```python
from lgos_openwebui.settings import Settings

settings = Settings()
auth = page.request.post(
    settings.URL + "/api/v1/auths/signin",
    data={"email": settings.ADMIN_EMAIL, "password": settings.ADMIN_PASSWORD},
)
assert auth.ok
page.goto(settings.URL, wait_until="domcontentloaded")
page.evaluate("token => localStorage.setItem('token', token)", auth.json()["token"])
page.goto(settings.URL + "/?model=lgos.lgos-a/persistent-plot-agent")
page.locator("#chat-input").fill("Show the quarterly revenue chart.")
page.locator("#chat-input").press("Enter")
```

Use the Workspace Model ID shown in `/api/models` for other graphs. Inspect
Plotly inside the message's iframe with `page.frames` or a
[frame locator](https://playwright.dev/python/docs/frames).

## Chainlit

Open the Chainlit URL. With the default mock login, fill `input[name=email]`
and `input[name=password]` with any nonempty demo values, then click the
`Sign In` button. This uses the shared `demo-user`; OAuth deployments require
their configured login flow.

Click the current profile label, then select `lgos-a/persistent-plot-agent`
with `page.get_by_text(..., exact=True)`. Wait for the profile settings reload
and chat input readiness before submitting: switching profiles resets the
session and can discard a message sent immediately. Fill `#chat-input` and
press Enter. Native Plotly elements render in the main page.

## Verify and diagnose

- Wait for `.js-plotly-plot` and the completed assistant reply. Inspect chart
  data with `locator.evaluate("el => el.data")`, try a hover or zoom, and capture
  and inspect a screenshot. In Open WebUI, also check the containing iframe's
  height: a chart may exist but be clipped. Reload the chat to check persistence.
- Save screenshots under `/tmp` and report the chat URL and checks performed.
  On failure, inspect visible errors, browser console/network failures, and
  logs for the affected service.
- Chainlit `provider query parameter is required`: generated-file downloads
  must use `files_request()` and its provider, like uploads already do.
- Open WebUI `Model not found`: check that the `generic` Function is active
  and the Workspace Model's base model exists. A failed Function import can
  disable it; a successful sync preserves that disabled state. Re-enable it
  when repairing that failure. See
  [Open WebUI Functions](https://docs.openwebui.com/features/extensibility/plugin/functions/).
- A clipped Open WebUI chart needs the native `iframe:height` notification
  supported by
  [FullHeightIframe](https://github.com/open-webui/open-webui/blob/main/src/lib/components/common/FullHeightIframe.svelte).
