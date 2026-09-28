# Events And Citations

Two deterministic graphs show how graph output crosses the OpenAI boundary
without turning UI notifications into tool calls.

| Graph | Public output | Client behavior |
| --- | --- | --- |
| `citation-events` | Markdown links and inline markers plus OpenAI `url_citation` annotations | Chainlit renders the Markdown; Open WebUI resolves markers through native source events |
| `status-events` | Standard assistant text plus Responses commentary | Chainlit uses a `TaskList`; Open WebUI persists native status history |

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

## Request Flow

1. A maintained demo UI sends a standard request through LGOS
   `/v1/responses` to `citation-events` or `status-events`.
2. `citation-events` returns portable Markdown plus standard OpenAI
   `url_citation` annotations.
3. `status-events` publishes three plain statuses through LangGraph's stream
   writer while returning ordinary assistant text.
4. LGOS always transports the assistant text. Streaming Responses translate
   status updates into standard `phase="commentary"` message items; no
   metadata opt-in is required. Chat Completions streams plain text deltas.
5. UI adapters render the fields and events they support. Other OpenAI clients
   can ignore the optional output and keep the text.

These graphs have no checkpointer or Store. Their graph state and event
timeline last for one request; each UI separately owns its transcript and
rendered status or activity history.

## Try It

| Model | Prompt | Transport |
| --- | --- | --- |
| `citation-events` | `Show me a cited answer.` | Responses |
| `status-events` | `Prepare the media workflow.` | Responses |

See [Citation Ownership](../../explanation/openai-compatibility.md#citation-ownership)
and [Streaming Status](../../explanation/openai-compatibility.md#streaming-status)
for the normative transport contract.
