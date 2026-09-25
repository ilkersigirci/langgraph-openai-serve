# Demo Design Choices

Why the demo works the way it does. Each section owns its component's
decisions. Add a row for each significant decision; when one changes, update
its row. Package decisions live in
[Design Choices](../explanation/design-choices.md).

## OpenTelemetry

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| The Collector normalizes `session.id` to `gen_ai.conversation.id` for API and worker spans; conversation tables select `lgos.graph_run`, and graph latency metrics include both services. | Both execution modes use the GenAI conversation attribute. Background submission finishes before execution and has no graph conversation attributes on its HTTP span. | Rows require the Langfuse callback, ingestion mapping, and a finished graph span; durations exclude Hatchet queue time. Older worker traces retain only `session.id`. | Graph spans supply `gen_ai.conversation.id` directly or dashboards need queue and submission latency. |
| The API and background worker use Hatchet's native instrumentor with the provider configured by `opentelemetry-instrument`; direct Hatchet collector export is disabled. | Native spans and trace propagation join background execution to the request through the existing Collector. SDK exclusions omit payloads and caller metadata. | Traces go to the configured observability backend; deployments must keep the native exclusions configured. | The deployment needs application spans in Hatchet's own trace viewer. |

## Gateways

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| Both UIs reach LGOS only through the gateway selected by `OPENAI_GATEWAY_TYPE`; neither connects to an API container or imports LGOS. | The demo exercises a real OpenAI-compatible edge, and UI inference cannot bypass the gateway's data plane. | Gateways normalize some metadata: error `type`, `param`, and `code`, and Bifrost's model detail. | A gateway preserves LGOS metadata and errors unchanged. |

### LiteLLM

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| Run the `homeserver-litellm` image and sync LGOS catalogs into native `model_info.lgos`. | The image preserves native Responses streaming and background lifecycles; the sync gives both UIs LGOS metadata through `/model/info`. | A custom image to maintain, and a sync to rerun after graph changes. | Upstream LiteLLM preserves streaming and background Responses. |

### Bifrost

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| One custom provider per API, a dedicated `lgos-files` provider, and a standard `openai` provider pinned to API A. | Separate identities show independent APIs behind one endpoint; normalized Files need their own provider; ID-only background retrieve and cancel need a standard provider. | More provider configuration, and model detail still needs pass-through. | Bifrost routes ID-only Responses calls to custom providers. |

## Chainlit

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| Human review is a persisted message whose custom form element submits through `callAction`. | Reload and navigation restore reviews from history, with no waiter, resume task, or session cache. `Ask*Message` blocks on its WebSocket, and Chainlit 2.12 does not persist plain `cl.Action` controls. | Custom JSX; a process-local submission lock; continuation and persistence are not transactional; a running response is not rebound to a new WebSocket, so returning early may need a refresh. | Chainlit persists actions or resumable asks, or the demo runs several Chainlit workers. |
| `HitlWorkflow` normalizes `createdAt` before updating a restored ledger. | Chainlit 2.12's PostgreSQL layer rejects the timestamp format it hydrates; without this, a ghost `pending` ledger blocks the next turn. | A workaround coupled to Chainlit internals. | After every Chainlit upgrade. |

## Open WebUI

| Choice | Why | Cost | Revisit when |
| --- | --- | --- | --- |
| One manifold Pipe serves every graph, and sync generates a Workspace Model per LGOS model from native `meta.chat_variables_schema`. | Each model gets Open WebUI's native settings form; JSON booleans survive, and UI settings never become prompt content. | The form is a generated projection tied to Open WebUI v0.11.3: rerun the sync after an LGOS schema change. | Open WebUI fetches a model's settings schema itself, or the image pin changes. |
| Open WebUI keeps its own raw upload copy; the central Files API owns the inference copy. | Open WebUI's native attachment UI requires its own file record. | Every upload is stored twice. | Open WebUI can attach an external file ID. |
