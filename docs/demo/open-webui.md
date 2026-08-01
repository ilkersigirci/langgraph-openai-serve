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
bundled Functions, reads the combined Bifrost model catalog, retrieves
each model's detailed LGOS metadata, hides the corresponding public manifold
base, and bulk-imports the generated Workspace Models. Run it again after
changing the Function, graph catalog, or a graph's client settings schema.

The operation is additive: it does not delete user-managed Functions or
Workspace Models. Generated manifold bases remain public and hidden so regular
users can access their visible wrappers without seeing duplicate selector
entries. The sync owns that base visibility and public access. New generated
Workspace Models are public; later syncs preserve their access grants and
active state. Their generated name, base model, settings schema, and model
parameters remain sync-owned. Delete generated records manually after removing
an LGOS model from the catalog.

The command discovers every top-level `.py` file in that directory except files
whose names start with `_`. The filename stem is the Function ID, and the
required Open WebUI frontmatter `title` is its display name. Function
filenames must be lowercase Python identifiers.

The defaults match `compose.yaml`. Override the Open WebUI connection with
`DEMO_OPENWEBUI_URL`, `DEMO_OPENWEBUI_ADMIN_EMAIL`, and
`DEMO_OPENWEBUI_ADMIN_PASSWORD`. Override LGOS discovery with
`DEMO_OPENWEBUI_CATALOG_BASE_URL`, `DEMO_OPENWEBUI_INFERENCE_BASE_URL`, and
`DEMO_OPENWEBUI_API_KEY`. Set secrets in the environment rather than passing
them on the command line.

Choose a generated entry such as `LGOS / lgos-a/simple-graph` to use Chat
Variables. Its Workspace Model ID is `lgos.lgos-a/simple-graph`, and its base
model is `generic.lgos-a/simple-graph`. The raw `Generic / ...` manifold entry
is public but hidden, following Open WebUI's
[curated-interface guidance](https://docs.openwebui.com/features/workspace/models/#recommended-a-hidden-public-base-model-with-a-curated-model-on-top).
`UserValves-Simple / simple-graph` remains available as the static alternative.

Configure the two in-container LGOS URLs and key independently in the generic
Function's admin valves. Open WebUI stores Function code in its database, so a
bind mount of the Python file does not update it.

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

The manifold Pipe opts into LGOS client stream events and maps every portable
status update to Open WebUI's native
[`status` events](https://docs.openwebui.com/features/extensibility/plugin/development/events/#status).
Select `status-events` and ask **Prepare the media workflow.** Open WebUI saves
each update in the assistant message's `statusHistory`; `done=False` displays an
active shimmer, `done=True` stops it, and `hidden=True` keeps the history entry
out of the current display. Persisted statuses survive a reload or closed tab.
`progress` and `artifact` events are currently ignored by this Pipe.

The adapter deliberately does not turn status updates into OpenAI tool calls.
Open WebUI treats a tool call as work it must execute, but LGOS has already
started the backend work. The passive status mapping keeps execution in the
graph and avoids an unknown-tool or duplicate-execution path.

When the Pipe targets an OpenAI-compatible proxy, status updates require a raw
pass-through inference URL because a schema-normalizing route may discard the
extension-only chunks. See
[proxy compatibility](../how-to-guides/openai-proxies.md#client-event-compatibility).

## Interrupt Approval

Select `interruptible-approval` from the manifold Pipe to try confirmation. The
Pipe sends `metadata.langgraph_thread_id`, presents the interrupt, and returns
the matching tool result when the user approves or rejects it.

See the core [citation contract](../explanation/openai-compatibility.md#citation-ownership)
and [interrupt protocol](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the API behavior beneath the adapter.
