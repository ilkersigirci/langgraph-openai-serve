# Events And Citations

Three deterministic graphs show how graph output crosses the OpenAI boundary
without turning UI notifications into tool calls.

| Graph | Public output | Client behavior |
| --- | --- | --- |
| `citation-events` | Markdown content plus OpenAI `url_citation` annotations | Chainlit renders the Markdown; Open WebUI also forwards streaming annotations |
| `status-events` | Standard assistant text plus opt-in `status` events | Chainlit uses a `TaskList`; Open WebUI persists native status history |
| `custom-event-showcase` | Assistant text interleaved with `progress` and `artifact` events | Chainlit renders its custom activity panel; Open WebUI ignores these unsupported kinds |

```mermaid
flowchart LR
  citations["citation-events"]
  statuses["status-events"]
  custom["custom-event-showcase"]
  lgos["LGOS stream adapter"]

  content["OpenAI delta.content"]
  annotations["OpenAI url_citation annotations"]
  events["Opt-in LGOS event chunks"]

  chainlit["Chainlit adapter"]
  openwebui["Open WebUI Pipe"]
  sdk["Standard OpenAI client"]

  citations -->|"text + citation_event"| lgos
  statuses -->|"text + status_event"| lgos
  custom -->|"text + client_event"| lgos

  lgos --> content
  lgos -->|"citation-events"| annotations
  lgos -->|"status and custom events"| events

  content --> chainlit
  content --> openwebui
  content --> sdk
  annotations --> openwebui
  annotations --> sdk
  events --> chainlit
  events --> openwebui
```

Assistant text remains portable. Citations use the standard OpenAI annotation
shape, while passive status, progress, and artifact updates use LGOS's
advertised `client_events` extension. A client that does not opt into or
understand that extension can still consume the assistant text.

## Try It

| Model | Prompt |
| --- | --- |
| `citation-events` | `Show me a cited answer.` |
| `status-events` | `Prepare the media workflow.` |
| `custom-event-showcase` | `Build the compatibility report.` |

See [Citation Ownership](../../explanation/openai-compatibility.md#citation-ownership)
and [Client Events](../../explanation/openai-compatibility.md#client-stream-events)
for the normative transport contract.
