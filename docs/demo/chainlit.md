# Chainlit Client

The included Chainlit UI is an optional OpenAI client of LGOS. It does not add
routes or change the server contract.

The Chainlit project intentionally does not install or import the
`langgraph-openai-serve` Python package. It demonstrates that a UI integration
needs only the OpenAI wire contract. Its local declarations cover only the LGOS
response extensions it consumes and link to their authoritative source files.

## Run The UI

Create the local environment file and a Chainlit signing secret:

```bash
cd demo
cp .env.example .env
uv run --directory ui/chainlit_ui --locked chainlit create-secret
```

Put the generated value in `CHAINLIT_AUTH_SECRET`. Chainlit also requires an
existing S3-compatible bucket: replace the example `BUCKET_NAME`,
`APP_AWS_ACCESS_KEY`, `APP_AWS_SECRET_KEY`, `APP_AWS_REGION`, and
`DEV_AWS_ENDPOINT` values before starting the UI.

=== "Compose"

    ```bash
    make run-chainlit
    ```

=== "Local processes"

    Start PostgreSQL and the API using
    [Run the Demo API](api.md#start-postgresql-and-the-api).

    In another terminal from `demo/`:

    ```bash
    make run-chainlit-local
    ```

Both modes apply pending Chainlit schema migrations before the UI starts. Open
`http://localhost:3002`. See [Docker Compose](docker.md#demo-services)
for container endpoints.

In direct mode, profile discovery reads each
`langgraph_openai_serve.description` from the model list. In Bifrost mode, the
catalog discovers providers and provider-qualified IDs, then one pass-through
list request per provider supplies the native model metadata. Listing,
retrieval, and chat use the same raw pass-through route and discovered
`x-model-provider`; the adapter has no provider list. The demo API owns the
descriptions. Chainlit keeps ordinary chat available but marks a model as
**Limited functionality** when an endpoint omits or strips one.

## Runtime Settings

After a profile is selected, Chainlit:

1. Retrieves the detailed model through the configured OpenAI client and reads
   `langgraph_openai_serve.client_settings`.
2. Renders supported JSON Schema properties as Chainlit Chat Settings.
3. Restores saved values that still match the supported widget type or choice.
4. Compares the selected values with the advertised defaults.
5. Sends changed values as JSON text in
   `metadata.langgraph_runtime_settings` on every completion.

Booleans become switches, inline string enums become selects, and strings
become text inputs. Other schema shapes are not rendered. The adapter checks
only boolean/string types and select membership when restoring the UI; it does
not interpret general JSON Schema constraints. LGOS remains the validation
authority. If the required LGOS model extension is unavailable, Chainlit hides
the controls, uses server defaults, and shows a transient **Limited
functionality** warning after selection. Profile discovery itself stays
list-only because descriptions arrive with the list response. Standard Chat
Completions remain available.

![Chainlit Settings panel showing conversation-history and audience controls](../static/runtime_settings_chainlit.png)

*Runtime settings discovered from `lgos-a/simple-graph` and rendered as native
Chainlit controls.*

The same panel includes a Chainlit-owned **Stream response** switch for every
profile. It defaults to enabled and controls the standard Chat Completions
`stream` parameter; it is not included in `langgraph_runtime_settings`. With
streaming disabled, Chainlit waits for the complete response and sends the
answer once. Transient client events remain streaming-only.

Chainlit may restore UI selections with a saved thread, but LGOS does not
persist runtime settings. The adapter resends non-default values for every
request that needs them. The underlying contract is documented in
[LangGraph Runtime Settings](../how-to-guides/langgraph-runtime-settings.md).

## Persistence And Login

Chainlit's PostgreSQL data layer stores users, threads, steps, and feedback.
Opening a stored thread restores its role/content transcript and continues with
the same login identity. The adapter also sends Chainlit's stable thread ID as
`metadata.session_id` on every completion, allowing Langfuse to group the
thread's per-request traces into one session. The `persistent-plot-agent` demo also
combines that value with the authenticated OpenAI `user` to scope its LangGraph
chart document. The transcript remains owned and resent by Chainlit; no chat
history is added to LGOS. See the
[persistent plot agent ownership flow](graphs/persistent-plot-agent.md#ownership-boundaries)
for the API Store, Chainlit PostgreSQL, and S3 boundaries.

=== "Mock login (default)"

    `DEMO_CHAINLIT_LOGIN_TYPE=mock` accepts any non-empty username and password
    and maps every session to the shared `demo-user`. This is for local use only.

=== "PocketID OAuth"

    Set `DEMO_CHAINLIT_LOGIN_TYPE=oauth` and provide the generic OAuth settings
    listed below. `OAUTH_GENERIC_USER_IDENTIFIER=sub` uses PocketID's stable
    subject as the Chainlit user identifier.

    Register `http://localhost:3002/auth/oauth/PocketID/callback` for local use.
    Behind a reverse proxy, set `CHAINLIT_URL` to the external HTTPS origin and
    register `${CHAINLIT_URL}/auth/oauth/${OAUTH_GENERIC_NAME}/callback`.

Browser login is separate from bearer-token protection for the LGOS `/v1` API.
See [Authentication](../how-to-guides/authentication.md).

## Interrupt Demo

Run the dedicated HITL UI:

```bash
DEMO_CHAINLIT_UI_FILE=hitl make run-chainlit-local
```

Initial requests need no interrupt metadata. The HITL client implements the
[canonical batch replay](../explanation/openai-compatibility.md#canonical-batch-replay):
it asks for every response, sends no partial batch, and repeats when the graph
pauses again. Each response is shown with Chainlit's native
[`AskElementMessage`](https://docs.chainlit.io/api-reference/ask/ask-for-element)
and a small custom element. Choice buttons and the allowed free-text field submit
one `{resume: ...}` value, so the client depends only on the standard tool-call
batch, not the graph topology. See the shared
[interrupt walkthrough](graphs/interruptible-approval.md).

!!! note "Reconnect recovery and its boundary"

    The adapter stores the exact assistant tool-call batch on the same
    model-context-excluded Chainlit message that displays the current prompt.
    Its
    [`on_chat_resume`](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume)
    hook restores the newest pending batch and reattaches its custom review form,
    including the free-text field when allowed, after the pinned Chainlit host
    hydrates the displayed thread. Refreshing abandons only the old live prompt;
    it neither duplicates the persisted message nor rejects or resumes the graph.
    Chainlit queues data-layer writes asynchronously, with no public flush API,
    so a process crash can still occur before that message reaches PostgreSQL.
    Once stored, cancellation, reload, or worker loss before the resume request
    does not require API-side chat history.

    The demo does not durably cache a terminal response or a later interrupt
    response that has not yet reached Chainlit. If the API accepts a resume but
    the worker loses the following response, replaying the older ledger fails
    safely as stale; the completed output or newer batch cannot be reconstructed
    from that old ledger. Applications requiring recovery across that window
    need a durable result/pending-response handoff in their UI boundary. See
    [Interruptible Human Review](graphs/interruptible-approval.md#postgresql-runtime)
    for server-side checkpoint retention.

## Streaming, Events, And Citations

Clicking **Stop** closes the OpenAI stream. Partial assistant text remains
visible but is excluded from later model context because it is incomplete.

The UI renders Markdown links, images, and inline citation markers from
assistant content. It does not consume structured OpenAI citation annotations.
The bundled adapter opts into LGOS client stream events only when model
retrieval advertises `client_events`. Portable status updates render as a
native Chainlit
[`TaskList`](https://docs.chainlit.io/api-reference/elements/tasklist). Other
events render as one live-updating Chainlit
[custom element](https://docs.chainlit.io/api-reference/elements/custom) per
completion. The panel shows event type, namespace, progress, and artifact
details, with a JSON fallback for other payload shapes. Its host message is
excluded from model context. A versioned `kind=chart` artifact instead renders
with Chainlit's native
[`Plotly`](https://docs.chainlit.io/api-reference/elements/plotly) element.
The adapter builds the Plotly figure from the event's small semantic payload;
the graph does not stream Plotly JSON. The official data layer stores the
element metadata in PostgreSQL and figure JSON in the configured S3-compatible
bucket, allowing the native chart to return with the thread. The packaged
botocore configuration uses Signature V4 and path-style addressing for
S3-compatible endpoints. Unknown extension versions are ignored.

Status task lists are live UI state and are not restored from persisted chat
history. Shared prompts and graph behavior are documented under
[Events And Citations](graphs/events-and-citations.md#try-it) and
[Persistent Plot Agent](graphs/persistent-plot-agent.md#try-it). A schema-normalizing proxy
may strip capability metadata and event-only chunks; see
[proxy compatibility](../how-to-guides/openai-proxies.md#client-event-compatibility).

## Settings Reference

LGOS endpoint settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `DEMO_CHAINLIT_OPENAI__BASE_URL` | `http://localhost:3004/v1` | Endpoint used for retrieval and inference. Direct mode also lists from it. |
| `DEMO_CHAINLIT_OPENAI__CATALOG_BASE_URL` | unset | Optional Bifrost model catalog endpoint; setting it enables provider-qualified pass-through routing. |
| `DEMO_CHAINLIT_OPENAI__API_KEY` | `DUMMY` | OpenAI API or gateway key. |
| `DEMO_CHAINLIT_HITL_MODEL` | `interruptible-approval` | Model selected by the HITL UI. |
| `DEMO_CHAINLIT_UI_FILE` | `simple` | Chainlit target: `simple` or `hitl`. |
| `DEMO_CHAINLIT_LOGIN_TYPE` | `mock` | Browser login: `mock` or `oauth`. |
| `CHAINLIT_UTILS_MIGRATIONS_TABLE` | `_lgos_chainlit_schema_migrations` | Migration ledger retained for existing demo databases. |
| `CHAINLIT_UTILS_MODEL_CONTEXT_EXCLUDED_KEY` | `lgos_chainlit.exclude_from_model_context` | Persisted metadata key for UI-only messages. |

See the bundled [Bifrost gateway](bifrost.md) for the Compose endpoint and
adapter behavior.

Native Chainlit settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | required | PostgreSQL data-layer URL. |
| `CHAINLIT_AUTH_SECRET` | required | Browser-session signing secret. |
| `CHAINLIT_APP_ROOT` | `src/lgos_chainlit` | Tracked UI configuration and welcome Markdown. |
| `BUCKET_NAME` | required | S3-compatible bucket for native elements. |
| `APP_AWS_ACCESS_KEY` | required | S3 access key. |
| `APP_AWS_SECRET_KEY` | required | S3 secret key. |
| `APP_AWS_REGION` | required | S3 signing region. |
| `DEV_AWS_ENDPOINT` | required | Custom S3-compatible endpoint URL. |
| `STORAGE_EXPIRY_TIME` | `3600` | Lifetime in seconds for resumed element URLs. |
| `CHAINLIT_URL` | request origin | External origin for OAuth callbacks. |
| `OAUTH_GENERIC_CLIENT_ID` | required for `oauth` | OAuth client ID. |
| `OAUTH_GENERIC_CLIENT_SECRET` | required for `oauth` | OAuth client secret. |
| `OAUTH_GENERIC_AUTH_URL` | required for `oauth` | Authorization endpoint. |
| `OAUTH_GENERIC_TOKEN_URL` | required for `oauth` | Token endpoint. |
| `OAUTH_GENERIC_USER_INFO_URL` | required for `oauth` | User-info endpoint. |
| `OAUTH_GENERIC_SCOPES` | required for `oauth` | Space-separated scopes. |
| `OAUTH_GENERIC_NAME` | `generic` | Provider ID used in the callback path. |
| `OAUTH_GENERIC_USER_IDENTIFIER` | `email` | User identifier claim. |

The element bucket must allow browser CORS `GET` and `HEAD` requests from the
Chainlit origin. CORS only permits the cross-origin response; the object still
requires Chainlit's time-limited presigned URL. See
[Amazon S3's CORS guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html).

The demo requires Chainlit 2.11.1 or newer. Review Chainlit's migration guidance
when updating it because the PostgreSQL schema is release-specific.

## Production Notes

- Use OAuth or another real callback; mock mode provides no access control or
  user isolation.
- Keep OAuth, signing, and object-storage secrets outside source control.
- Restrict `allow_origins` to the deployed HTTPS origin.
- Configure session affinity for multiple UI workers. The demo keeps user file
  uploads disabled but requires object storage for native Plotly persistence.
- Run `lgos-chainlit-setup` before starting or replacing workers.

See Chainlit's documentation for
[password callbacks](https://docs.chainlit.io/authentication/password),
[OAuth](https://docs.chainlit.io/authentication/oauth),
[PostgreSQL persistence](https://docs.chainlit.io/data-layers/official), and
[deployment](https://docs.chainlit.io/deploy/overview).
