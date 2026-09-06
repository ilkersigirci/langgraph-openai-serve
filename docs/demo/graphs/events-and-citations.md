# Events And Citations

Three deterministic graphs show how graph output crosses the OpenAI boundary
without turning UI notifications into tool calls.

| Graph | Public output | Client behavior |
| --- | --- | --- |
| `citation-events` | Markdown links and inline markers plus OpenAI `url_citation` annotations | Chainlit renders the Markdown; Open WebUI resolves markers through native source events |
| `status-events` | Standard assistant text plus Responses commentary | Chainlit uses a `TaskList`; Open WebUI persists native status history |
| `custom-event-showcase` | Standard assistant text | Responses returns the final text; Chat Completions streams plain text |

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

1. A maintained demo UI sends a standard Responses request to `citation-events`
   or `status-events`.
2. `citation-events` returns portable Markdown plus standard OpenAI
   `url_citation` annotations.
3. `status-events` publishes passive updates through LangGraph's stream writer
   while returning ordinary assistant text.
4. LGOS always transports the assistant text. Streaming Responses translate
   visible status updates into standard `phase="commentary"` message items; no
   metadata opt-in is required. Chat Completions streams plain text deltas.
5. UI adapters render the fields and events they support. Other OpenAI clients
   can ignore the optional output and keep the text.

`custom-event-showcase` demonstrates how graphs emitting internal stream writer
events behave across protocols. When streamed via Responses or Chat Completions,
the API boundary safely discards non-commentary custom events and delivers the
standard assistant text without requiring ad-hoc envelopes.

These graphs have no checkpointer or Store. Their graph state and event
timeline last for one request; each UI separately owns its transcript and
rendered status or activity history.

## Try It

| Model | Prompt | Transport |
| --- | --- | --- |
| `citation-events` | `Show me a cited answer.` | Responses |
| `status-events` | `Prepare the media workflow.` | Responses |
| `custom-event-showcase` | `Build the compatibility report.` | Responses or Chat stream |

See [Citation Ownership](../../explanation/openai-compatibility.md#citation-ownership)
and [Streaming Status](../../explanation/openai-compatibility.md#streaming-status)
for the normative transport contract.
