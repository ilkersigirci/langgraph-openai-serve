# Open WebUI Functions

The demo includes two Open WebUI Functions over the LGOS OpenAI-compatible API:

- `demo/ui/openwebui/src/lgos_openwebui/functions/generic.py` is a
  [manifold Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/#creating-multiple-models-with-pipes)
  for all registered graphs. It handles streaming, citations, and interrupt
  approval, and forwards graph-specific runtime settings.
- `demo/ui/openwebui/src/lgos_openwebui/functions/uservalves_simple.py` keeps
  the earlier static `UserValves` design as a small single-model example.

The sync command also generates one Open WebUI Workspace Model per discovered
LGOS model. Each Workspace Model wraps the corresponding manifold model and
projects its LGOS settings schema into native
[Chat Variables](https://docs.openwebui.com/features/chat-conversations/chat-features/chat-params/#chat-variables).

## Setup

Start the official Open WebUI image:

```bash
cd demo
cp .env.example .env
docker compose -f compose.yaml up --wait open-webui
```

Then run the independent synchronization project locally:

```bash
make sync-openwebui
```

The sync command signs in through `/api/v1/auths/signin`, creates or updates the
bundled Functions, lists and retrieves LGOS models through one OpenAI client,
hides the corresponding public manifold base, and bulk-imports the generated
Workspace Models. Run it again after changing the Function, graph catalog, or a
graph's client settings schema.

Generated Workspace Model descriptions come from the selected graph's required
`GraphConfig.description`. The sync marks a model as **Limited functionality**
when the API omits a description.

The operation is additive: it does not delete user-managed Functions or
Workspace Models. Generated manifold bases remain public and hidden so regular
users can access their visible wrappers without seeing duplicate selector
entries. The sync owns that base visibility and public access. New generated
Workspace Models are public; later syncs preserve their access grants and
active state. Their generated name, base model, settings schema, description,
and parameters remain sync-owned. Delete generated records manually after
removing an LGOS model from the catalog.

The command discovers every top-level `.py` file in that directory except files
whose names start with `_`. The filename stem is the Function ID, and the
required Open WebUI frontmatter `title` is its display name. Function
filenames must be lowercase Python identifiers.

The defaults match `compose.yaml`. Override the Open WebUI connection with
`DEMO_OPENWEBUI_URL`, `DEMO_OPENWEBUI_ADMIN_EMAIL`, and
`DEMO_OPENWEBUI_ADMIN_PASSWORD`. Configure the one LGOS client with
`DEMO_OPENWEBUI_OPENAI_BASE_URL`, `DEMO_OPENWEBUI_API_KEY`, and
`DEMO_OPENWEBUI_MODEL_ROUTES`. The last setting is a JSON object mapping
synthetic model prefixes to request headers. The demo maps `lgos-a` and
`lgos-b` to Bifrost's corresponding `x-model-provider` values. Set it to `{}`
for a standard endpoint whose listed model IDs should be reused verbatim. Set
secrets in the environment rather than passing them on the command line. Point
this setting and the Function valve below at the same deployment; their
hostnames differ when one runs on the host and the other runs inside Compose.

Choose a generated entry such as `LGOS / lgos-a/simple-graph` to use Chat
Variables. Its Workspace Model ID is `lgos.lgos-a/simple-graph`, and its base
model is `generic.lgos-a/simple-graph`. The raw `Generic / ...` manifold entry
is public but hidden, following Open WebUI's
[curated-interface guidance](https://docs.openwebui.com/features/workspace/models/#recommended-a-hidden-public-base-model-with-a-curated-model-on-top).
`UserValves-Simple / simple-graph` remains available as the static alternative.

Configure `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, and
`OPENAI_API_MODEL_ROUTES` in the generic Function's admin valves. A configured
route adds its prefix to the selector and applies its headers to listing,
detailed retrieval, and inference. Leave the route object empty for a standard
OpenAI endpoint. The static UserValves Function instead accepts one `MODEL` and
one `OPENAI_API_HEADERS` object and sends both unchanged. Open WebUI stores
Function code in its database, so a bind mount of the Python file does not
update it.

The generic selector uses only model listing and labels entries without the
required LGOS description as **Limited functionality**. It retrieves detailed
metadata after a model is selected for chat, when settings and capability
checks need it.

## Limited Functionality

Every generated model remains visible when its detailed response lacks the
required `langgraph_openai_serve` extension. Its name and description say
**Limited functionality**. At chat time both bundled Pipes also emit an Open WebUI
[`notification`](https://docs.openwebui.com/features/extensibility/plugin/development/events/#notification)
with warning severity. Standard assistant text may still work; runtime settings,
client events, and interrupts are not assumed. Configure the selected OpenAI
URL as the proxy's LGOS pass-through route to remove the warning.

## Runtime Settings

LGOS remains the schema and default-value source of truth. The sync command
uses the same deliberately small JSON Schema subset as the Chainlit demo:

- boolean with a boolean default becomes a checkbox;
- string enum with a valid string default becomes a selector;
- string with a string default becomes a text input;
- nested objects, arrays, numbers, and unsupported schemas are omitted.

Open WebUI stores Chat Variable values on the conversation. Select a generated
LGOS model, then use the Chat Variables control beside the message input. Since
LGOS supplies defaults for every setting, the form does not block the first
message merely to confirm them.

When a chat has values, the Pipe retrieves the selected model's current LGOS
metadata, ignores names no longer present, removes values equal to current
defaults, and sends only changes as
`metadata.langgraph_runtime_settings`. LGOS performs the authoritative runtime
validation.

The Workspace Model schema is a generated projection, not a second
configuration source. Open WebUI does not fetch a remote schema when the model
selector changes, so rerun `make sync-openwebui` after an LGOS schema change.
Model selection then switches among the already-synchronized native forms.

### Static Alternative

`uservalves_simple.py` remains useful for one fixed graph when settings are
intentionally hand-maintained per user and dynamic discovery or per-chat values
are unnecessary. Its fields are illustrative; generated Workspace Models still
take their schemas only from LGOS.

!!! note "Pinned Open WebUI contract"

    The demo pins Open WebUI v0.11.0. The sync imports its native
    `meta.chat_variables_schema` model metadata directly instead of putting
    form declarations in a system prompt. This preserves JSON booleans and
    ensures UI configuration never becomes graph prompt content. Recheck this
    integration contract when changing the Open WebUI pin. The pinned source
    keeps extra [Workspace Model metadata](https://github.com/open-webui/open-webui/blob/f9590b8017199e56d5e953657e6498e3cef1d246/backend/open_webui/models/models.py#L67-L76)
    and reads the schema in the
    [Chat Variables UI](https://github.com/open-webui/open-webui/blob/f9590b8017199e56d5e953657e6498e3cef1d246/src/lib/components/chat/Chat.svelte#L396-L405).

## Streaming, Status, And Citations

The general manifold Pipe streams assistant content unchanged, so Open WebUI
renders Markdown links and images normally. For streaming requests it also
forwards final OpenAI citation annotations without translating them.
Non-streaming generator results remain plain text. The static example streams
assistant text only.

The manifold Pipe opts into LGOS client stream events only when model retrieval
advertises `client_events`, and maps every portable status update to Open
WebUI's native
[`status` events](https://docs.openwebui.com/features/extensibility/plugin/development/events/#status).
Select `lgos-a/status-events` and ask **Prepare the media workflow.** Open WebUI
saves each update in the assistant message's `statusHistory`; `done=False`
displays an active shimmer, `done=True` stops it, and `hidden=True` keeps the
history entry out of the current display. Persisted statuses survive a reload
or closed tab. `progress` and `artifact` events are currently ignored by this
Pipe.

The adapter deliberately does not turn status updates into OpenAI tool calls.
Open WebUI treats a tool call as work it must execute, but LGOS has already
started the backend work. The passive status mapping keeps execution in the
graph and avoids an unknown-tool or duplicate-execution path.

When the Pipe targets an OpenAI-compatible proxy, its one OpenAI base URL must
be a raw pass-through because a schema-normalizing route may discard model
metadata and extension-only chunks. For a multiplexed pass-through, configure
its model routes and headers explicitly. For a normalized endpoint, clear the
route object so returned model IDs are reused verbatim; standard chat may work,
but the Pipe remains in limited mode when LGOS metadata is missing. See
[proxy compatibility](../how-to-guides/openai-proxies.md#client-event-compatibility).

## Interrupt Approval

Select `lgos-b/interruptible-approval` from the manifold Pipe to try
confirmation. The Pipe sends `metadata.langgraph_thread_id`, presents the
interrupt, and returns the matching tool result when the user approves or
rejects it.

See the core [citation contract](../explanation/openai-compatibility.md#citation-ownership)
and [interrupt protocol](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the API behavior beneath the adapter.
