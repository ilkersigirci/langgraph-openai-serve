# Events And Citations

Three deterministic graphs show how graph output crosses the OpenAI boundary
without turning UI notifications into tool calls.

| Graph | Public output | Client behavior |
| --- | --- | --- |
| `citation-events` | Markdown links and inline markers plus OpenAI `url_citation` annotations | Chainlit renders the Markdown; Open WebUI resolves markers through native source events |
| `status-events` | Standard assistant text plus opt-in `status` events | Chainlit uses a `TaskList`; Open WebUI persists native status history |
| `custom-event-showcase` | Assistant text interleaved with `progress` and `artifact` events | Chainlit renders its custom activity panel; Open WebUI ignores these unsupported kinds |

## LangGraph Topology

=== "citation-events"

    ```mermaid
    graph TD;
		__start__ --> answer_with_citation;
		answer_with_citation --> __end__;
    ```

=== "status-events"

    ```mermaid
    graph TD;
		__start__ --> prepare_media;
		prepare_media --> __end__;
    ```

=== "custom-event-showcase"

    ```mermaid
    graph TD;
		__start__ --> build_compatibility_report;
		build_compatibility_report --> __end__;
    ```

## Request Flow

1. The client sends a standard Chat Completions request to one deterministic
   graph.
2. `citation-events` returns portable Markdown plus standard OpenAI
   `url_citation` annotations.
3. `status-events` and `custom-event-showcase` publish passive updates through
   LangGraph's stream writer while returning ordinary assistant text.
4. LGOS always transports the assistant text. It includes client-event chunks
   only for streaming requests with
   `metadata.langgraph_stream_events="v1"`.
5. UI adapters render the fields and events they support. Other OpenAI clients
   can ignore the optional output and keep the text.

These graphs have no checkpointer or Store. Their graph state and event
timeline last for one request; each UI separately owns its transcript and
rendered status or activity history.

## Try It

| Model | Prompt |
| --- | --- |
| `citation-events` | `Show me a cited answer.` |
| `status-events` | `Prepare the media workflow.` |
| `custom-event-showcase` | `Build the compatibility report.` |

See [Citation Ownership](../../explanation/openai-compatibility.md#citation-ownership)
and [Client Events](../../explanation/openai-compatibility.md#client-stream-events)
for the normative transport contract.
