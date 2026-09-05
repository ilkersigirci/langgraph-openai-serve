# Events And Citations

Three deterministic graphs show how graph output crosses the OpenAI boundary
without turning UI notifications into tool calls.

| Graph | Public output | Client behavior |
| --- | --- | --- |
| `citation-events` | Markdown links and inline markers plus OpenAI `url_citation` annotations | Chainlit renders the Markdown; Open WebUI resolves markers through native source events |
| `status-events` | Standard assistant text plus Responses commentary (or opted-in direct Chat status events) | Chainlit uses a `TaskList`; Open WebUI persists native status history |
| `custom-event-showcase` | Standard assistant text plus opted-in direct Chat `progress` and `artifact` events | Direct OpenAI SDK clients inspect the namespaced extension; Responses returns only the final text |

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
   metadata opt-in is required. Direct Chat clients use the separate v1
   client-event opt-in.
5. UI adapters render the fields and events they support. Other OpenAI clients
   can ignore the optional output and keep the text.

`custom-event-showcase` is intentionally a direct Chat diagnostic, not a
maintained UI flow. It interleaves three small progress payloads with assistant
text and then emits an artifact descriptor. An OpenAI SDK client opts in and
reads the namespaced property from `model_extra`:

```python
stream = client.chat.completions.create(
    model="custom-event-showcase",
    messages=[{"role": "user", "content": "Build the compatibility report."}],
    stream=True,
    metadata={"langgraph_stream_events": "v1"},
)

for chunk in stream:
    extension = (chunk.model_extra or {}).get("langgraph_openai_serve")
    if extension:
        print(extension["event"])
```

Connect this client directly to LGOS. The property is not part of the standard
Chat schema and a normalizing proxy may remove it. The same graph remains a
valid Responses model, but Responses deliberately ignores its progress and
artifact events and returns only the standard final answer.

These graphs have no checkpointer or Store. Their graph state and event
timeline last for one request; each UI separately owns its transcript and
rendered status or activity history.

## Try It

| Model | Prompt | Transport |
| --- | --- | --- |
| `citation-events` | `Show me a cited answer.` | Responses |
| `status-events` | `Prepare the media workflow.` | Responses |
| `custom-event-showcase` | `Build the compatibility report.` | Direct Chat stream |

See [Citation Ownership](../../explanation/openai-compatibility.md#citation-ownership)
and [Streaming Status](../../explanation/openai-compatibility.md#streaming-status)
for the normative transport contract.
