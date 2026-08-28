# Complex Subgraphs

`complex-subgraphs` demonstrates a coordinator graph with specialist subgraphs
and one nested grandchild graph. The OpenAI request adapter supplies the final
user message as `question`. The keyword graph keeps its routing result in graph
state and emits an optional status update. The selected specialist writes the
final assistant message to the shared `messages` channel.

```mermaid
flowchart TD
  request([OpenAI user message]) --> adapter["request_to_input"]
  adapter --> route["Normalize question and select route"]

  subgraph api["API contract subgraph"]
    checks["Collect contract checks"] --> api_summary["Stream contract summary"]
  end

  subgraph docs["Docs specialist subgraph"]
    subgraph keywords["Keyword grandchild subgraph"]
      extract["Select keywords and emit status"]
    end
    extract --> docs_summary["Stream docs summary"]
  end

  route -->|"API, OpenAI, adapter, stream, or serve"| checks
  route -->|"other questions"| extract
  api_summary --> output["final assistant message"]
  docs_summary --> output
  output --> response([OpenAI assistant text])
```

LGOS exposes the answer-producing nested nodes `summarize_contract` and
`summarize_docs` as assistant text. The keyword node's selected terms are
available as graph state and, for clients that opt into `client_events`, as a
status update. The final assistant message is identical for streaming and
non-streaming requests.
