# LGOS Chainlit UI

Standalone Chainlit client for an OpenAI-compatible LGOS endpoint. It
intentionally does not install or import the `langgraph-openai-serve` Python
package, demonstrating that UI logic needs only the OpenAI wire protocol.

The client uses Responses exclusively and never connects directly to an LGOS
or Files container. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the gateway.
LiteLLM uses managed Responses routing; Bifrost uses its native Responses
route. Both use their normal Files and aggregate MCP routes with the same
gateway credential. LiteLLM discovery and settings read
`/model/info`, using `model_name` unchanged and the full `model_info.lgos`
extension. Bifrost uses its aggregate catalog and model-detail pass-through.
Before using independently started LiteLLM components, [sync the LGOS metadata](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/litellm-sync.md).
The full-stack `just demo/compose [--dev] [--otel]` variants do this
before starting Chainlit.

Before starting, replace the example signing secret and configure the required
S3-compatible bucket and credentials in `.env`.

The small set of LGOS-specific model-detail fields and metadata keys used by
this client is declared locally in
[`lgos_protocol.py`](src/lgos_chainlit/lgos_protocol.py). That file links every
declaration to its authoritative source in the main LGOS repository.

Use the [editable commands below](#local-utility-development) when testing
changes from a sibling `chainlit-utils` checkout.

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-chainlit-setup
uv run --locked --env-file .env lgos-chainlit
```

Application settings use the `DEMO_CHAINLIT_` prefix, except for the shared
gateway type, base URL, and API key. Reusable helper settings use `CHAINLIT_UTILS_`;
Chainlit's native `DATABASE_URL` and `CHAINLIT_AUTH_SECRET` variables remain
unprefixed. Native Chainlit elements use `BUCKET_NAME`, `APP_AWS_*`, and
`DEV_AWS_ENDPOINT` S3 settings so generated files survive thread resume.

`DEMO_CHAINLIT_LOGIN_TYPE=oauth` enables OIDC browser login independently of
gateway authorization. By default, mock and OAuth login both use
`OPENAI_GATEWAY_API_KEY` for Responses, Files, and MCP. Set
`DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING=true` to send the signed-in user's
access token instead. Delegated mode disables the static MCP connection and needs the gateway's
API permission and `offline_access`; point `OPENAI_GATEWAY_BASE_URL` at the
LiteLLM SSO endpoint. LiteLLM must authorize `/model/info` as well as Responses
and Files for the user's credential. See the
[Chainlit guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/chainlit.md#persistence-and-login)
for both modes, key rotation, and logout behavior.

Reusable OIDC routes, request credential isolation, and encrypted token storage
come from `chainlit-utils`. [`auth.py`](src/lgos_chainlit/auth.py)
maps this demo's settings and login policy onto those services. The wiring
coverage is in [`test_auth.py`](tests/test_auth.py).

Authlib handles OIDC discovery, S256 PKCE, state, nonce, and ID-token validation.
Set `DEMO_CHAINLIT_OAUTH_ISSUER` to the exact HTTPS issuer and `CHAINLIT_URL` to
the external HTTPS origin. Endpoints come from discovery; identity is the
verified `sub`.

`DEMO_CHAINLIT_OAUTH_CLIENT_AUTH_METHOD` defaults to `client_secret_basic`;
set `client_secret_post` if that is your registered client's method. The same
method is used for code exchange and, in delegated mode, refresh and revocation.
Set `DEMO_CHAINLIT_OAUTH_RESOURCE` when the provider uses RFC 8707 resource
indicators, such as PocketID's API resource. Otherwise configure the gateway
audience through the provider's client/scopes settings.
No provider names or access-token claim formats are built into Chainlit; the
gateway owns token validation.

Delegated mode requires `DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS`, a JSON list of
Fernet keys separate from `CHAINLIT_AUTH_SECRET`. Generate a key locally:

```bash
uv run python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

All workers share these encryption keys. When forwarding is enabled, PostgreSQL
holds a separate encrypted grant for each browser login and coordinates token
refresh. Chainlit owns UI authentication and history in either mode. Logout
removes any local gateway grant, not the identity provider's global session or
copied native UI JWTs. The guide above documents the lifecycle and security
boundaries.

Run the regular checks with `uv run --locked pytest`, or only the authentication
checks with `uv run --locked pytest tests/test_auth.py`. Use the editable
equivalents below when testing unpublished utility changes. The demo covers login
policy and gateway credential wiring. Reusable browser login, token encryption,
refresh concurrency, key rotation, and logout persistence are tested in the
[`chainlit-utils` repository](https://github.com/ilkersigirci/chainlit-utils).

## Local utility development

When compatible utility changes have not been published yet, use the sibling
`chainlit-utils` checkout as a temporary editable overlay:

```bash
uv run --locked --with-editable "../../../../chainlit-utils[sso]" \
  --env-file .env lgos-chainlit-setup
uv run --locked --with-editable "../../../../chainlit-utils[sso]" pytest
uv run --locked --with-editable "../../../../chainlit-utils[sso]" \
  ty check src --extra-search-path ../../../../chainlit-utils/src
uv run --locked --with-editable "../../../../chainlit-utils[sso]" \
  --env-file .env lgos-chainlit
```

The overlay keeps local paths out of the project manifest and lockfile. After
publishing, raise the demo's `chainlit-utils[sso]` minimum version to the release
that supplies the imported API and refresh `uv.lock`.

## Module ownership

`chat.py` registers the Chainlit callbacks. `auth.py` configures login and
gateway credentials; `clients.py` and
`gateway.py` own gateway access. `conversation.py`, `chat_settings.py`,
`files.py`, `display_files.py`, and `mcp.py` contain their respective
LGOS-specific integrations. `lgos_protocol.py` owns the LGOS wire declarations;
`interrupts.py` owns the LGOS interrupt payload and `InterruptReview` UI.

Import reusable history, settings, Responses, and durable HITL helpers
from their concrete modules under `chainlit_utils.chat` and
`chainlit_utils.openai`.

## Attachments

User attachments are uploaded separately through the selected gateway's normal
OpenAI Files API. LiteLLM assigns those requests to `litellm_proxy`; Bifrost
assigns them to `lgos-files`. The returned `file_id` reaches the graph as a
Responses `input_file` content part.
