# LGOS Chainlit UI

Standalone Chainlit client for an OpenAI-compatible LGOS endpoint. It
intentionally does not install or import the `langgraph-openai-serve` Python
package, demonstrating that UI logic needs only the OpenAI wire protocol.

The client uses Responses exclusively and never connects directly to an LGOS
or Files container. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the gateway.
LiteLLM uses managed Responses routing; Bifrost uses its native Responses
route. Both use their normal Files route. LiteLLM discovery and settings read
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

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-chainlit-setup
uv run --locked --env-file .env lgos-chainlit
```

Application settings use the `DEMO_CHAINLIT_` prefix, except for the shared
gateway type and base URL. Reusable helper settings use `CHAINLIT_UTILS_`;
Chainlit's native `DATABASE_URL` and `CHAINLIT_AUTH_SECRET` variables remain
unprefixed. Native Chainlit elements use `BUCKET_NAME`, `APP_AWS_*`, and
`DEV_AWS_ENDPOINT` S3 settings so generated files survive thread resume.

`DEMO_CHAINLIT_LOGIN_TYPE=oauth` enables OIDC browser login independently of
gateway authorization. By default, mock and OAuth login both use
`DEMO_CHAINLIT_GATEWAY_API_KEY`. Set
`DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING=true` and clear that key to send
the signed-in user's access token instead. Delegated mode needs the gateway's
API permission and `offline_access`; point `OPENAI_GATEWAY_BASE_URL` at the
LiteLLM SSO endpoint. LiteLLM must authorize `/model/info` as well as Responses
and Files for the user's credential. See the
[Chainlit guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/chainlit.md#persistence-and-login)
for both modes, key rotation, and logout behavior.

Authentication code lives in [`src/lgos_chainlit/auth/`](src/lgos_chainlit/auth/):
`chainlit.py` integrates login and request credentials with Chainlit,
`oauth_client.py` configures Authlib, and `oauth_tokens.py` stores and refreshes
encrypted grants. The corresponding tests are grouped in [`tests/auth/`](tests/auth/).

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
checks with `uv run --locked pytest tests/auth`. Delegated-token persistence and
refresh tests use a temporary schema in the configured PostgreSQL database:

```bash
TEST_CHAINLIT_DATABASE_URL=postgresql://lgos:lgos@localhost:3001/lgos \
  uv run --locked pytest -m integration tests/auth
```

`just demo/test-postgres --editable` runs these checks together with
the demo API's PostgreSQL tests. Login and request-isolation tests run in the
regular suite; neither suite needs a live identity provider.

User attachments are uploaded separately through the selected gateway's normal
OpenAI Files API. LiteLLM assigns those requests to `litellm_proxy`; Bifrost
assigns them to `lgos-files`. The returned `file_id` reaches the graph as a
Responses `input_file` content part.
