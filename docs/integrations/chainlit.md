# Chainlit

The included Chainlit UI is an optional OpenAI client of LGOS. It does not add
routes or change the server contract.

## Run The UI

Create the local environment file and a Chainlit signing secret:

```bash
cp .env.example .env
uv run chainlit create-secret
```

Put the generated value in `CHAINLIT_AUTH_SECRET`, then start PostgreSQL, the
demo API, and Chainlit in separate terminals:

```bash
docker compose up -d lgos-postgres
make run-demo-api
```

```bash
make run-demo-ui-chainlit
```

The UI target applies pending Chainlit schema migrations before starting. Open
`http://localhost:5000`. For the Compose service and container endpoints, see
[Docker](../how-to-guides/docker.md#demo-compose).

## Runtime Settings

After a profile is selected, Chainlit:

1. Retrieves the detailed model and reads
   `langgraph_openai_serve.client_settings`.
2. Renders supported JSON Schema properties as Chainlit Chat Settings.
3. Restores only saved values that remain valid under the current descriptor.
4. Compares the selected values with the advertised defaults.
5. Sends changed values as JSON text in
   `metadata.langgraph_runtime_settings` on every completion.

Booleans become switches, inline string enums become selects, and strings
become text inputs. Other schema shapes are not rendered. LGOS remains the
validation authority. If detailed model metadata is unavailable, Chainlit hides
the controls and uses server defaults.

Chainlit may restore UI selections with a saved thread, but LGOS does not
persist runtime settings. The adapter resends non-default values for every
request that needs them. The underlying contract is documented in
[LangGraph Runtime Settings](../how-to-guides/langgraph-runtime-settings.md).

## Persistence And Login

Chainlit's PostgreSQL data layer stores users, threads, steps, and feedback.
Opening a stored thread restores its role/content transcript and continues with
the same login identity.

=== "Mock login (default)"

    `DEMO_CHAINLIT_LOGIN_TYPE=mock` accepts any non-empty username and password
    and maps every session to the shared `demo-user`. This is for local use only.

=== "PocketID OAuth"

    Set `DEMO_CHAINLIT_LOGIN_TYPE=oauth` and provide the generic OAuth settings
    listed below. `OAUTH_GENERIC_USER_IDENTIFIER=sub` uses PocketID's stable
    subject as the Chainlit user identifier.

    Register `http://localhost:5000/auth/oauth/PocketID/callback` for local use.
    Behind a reverse proxy, set `CHAINLIT_URL` to the external HTTPS origin and
    register `${CHAINLIT_URL}/auth/oauth/${OAUTH_GENERIC_NAME}/callback`.

Browser login is separate from bearer-token protection for the LGOS `/v1` API.
See [Authentication](../how-to-guides/authentication.md).

## Interrupt Demo

Run the dedicated approval UI:

```bash
make run-demo-ui-chainlit-hitl
```

The HITL client adds the assistant tool call and matching tool result to the
immediate resume request. Chainlit's saved role/content transcript is not a
canonical tool-protocol ledger, so a future general tool-executing UI must store
completed tool pairs explicitly.

## Streaming, Events, And Citations

Clicking **Stop** closes the OpenAI stream. Partial assistant text remains
visible but is excluded from later model context because it is incomplete.

The UI renders Markdown links and images from assistant content. It does not
consume structured OpenAI citation annotations. The bundled adapter opts into
LGOS client stream events and renders them as one live-updating Chainlit
[custom element](https://docs.chainlit.io/api-reference/elements/custom) per
completion. The panel shows event type, namespace, progress, and artifact
details, with a JSON fallback for other payload shapes. Its host message is
excluded from model context. Unknown extension versions are ignored.

To see the demo, select `custom-event-showcase` and ask **Build the compatibility
report.** The activity panel advances from status through progress to an
artifact while the assistant answer streams independently.

Behind an OpenAI-compatible proxy, the activity panel requires a raw
pass-through inference URL. A schema-normalizing route may still stream the
answer while silently omitting event-only chunks. See
[OpenAI-Compatible Proxies](openai-proxies.md#client-event-compatibility).

## Settings Reference

LGOS endpoint settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `DEMO_CHAINLIT_INFERENCE__BASE_URL` | `http://localhost:8000/v1` | Inference API or gateway. |
| `DEMO_CHAINLIT_INFERENCE__API_KEY` | `DUMMY` | Inference API or gateway key. |
| `DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX` | empty | Optional proxy model namespace, such as `openai/`. |
| `DEMO_CHAINLIT_DISCOVERY__BASE_URL` | unset | Detailed discovery endpoint; otherwise inference is reused. |
| `DEMO_CHAINLIT_DISCOVERY__API_KEY` | unset | Required with an explicit discovery endpoint. |
| `DEMO_CHAINLIT_HITL_MODEL` | `interruptible-approval` | Model selected by the HITL UI. |
| `DEMO_CHAINLIT_UI_FILE` | `simple` | Chainlit target: `simple` or `hitl`. |
| `DEMO_CHAINLIT_LOGIN_TYPE` | `mock` | Browser login: `mock` or `oauth`. |

See [OpenAI-Compatible Proxies](openai-proxies.md) for separate inference and
discovery endpoint examples.

Native Chainlit settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | required | PostgreSQL data-layer URL. |
| `CHAINLIT_AUTH_SECRET` | required | Browser-session signing secret. |
| `CHAINLIT_APP_ROOT` | `demo/ui/chainlit_ui` in `.env.example` | Tracked UI configuration and welcome Markdown. |
| `CHAINLIT_URL` | request origin | External origin for OAuth callbacks. |
| `OAUTH_GENERIC_CLIENT_ID` | required for `oauth` | OAuth client ID. |
| `OAUTH_GENERIC_CLIENT_SECRET` | required for `oauth` | OAuth client secret. |
| `OAUTH_GENERIC_AUTH_URL` | required for `oauth` | Authorization endpoint. |
| `OAUTH_GENERIC_TOKEN_URL` | required for `oauth` | Token endpoint. |
| `OAUTH_GENERIC_USER_INFO_URL` | required for `oauth` | User-info endpoint. |
| `OAUTH_GENERIC_SCOPES` | required for `oauth` | Space-separated scopes. |
| `OAUTH_GENERIC_NAME` | `generic` | Provider ID used in the callback path. |
| `OAUTH_GENERIC_USER_IDENTIFIER` | `email` | User identifier claim. |

The demo requires Chainlit 2.11.1 or newer. Review Chainlit's migration guidance
when updating it because the PostgreSQL schema is release-specific.

## Production Notes

- Use OAuth or another real callback; mock mode provides no access control or
  user isolation.
- Keep OAuth and signing secrets outside source control.
- Restrict `allow_origins` to the deployed HTTPS origin.
- Configure session affinity for multiple UI workers and supported object
  storage before enabling file uploads.
- Run `uv run --module demo.ui.chainlit_ui.setup_database` before starting or
  replacing workers.

See Chainlit's documentation for
[password callbacks](https://docs.chainlit.io/authentication/password),
[OAuth](https://docs.chainlit.io/authentication/oauth),
[PostgreSQL persistence](https://docs.chainlit.io/data-layers/official), and
[deployment](https://docs.chainlit.io/deploy/overview).
