# Complex Subgraphs

`complex-subgraphs` demonstrates a coordinator graph with specialist subgraphs
and one nested grandchild graph. The OpenAI request adapter supplies the final
user message as `question`. The keyword graph keeps its selected terms in graph
state and emits an optional status update. The selected specialist writes the
final assistant message to the shared `messages` channel.

## LangGraph Topology

The native `xray` view expands the specialist and keyword subgraphs.

```mermaid
graph TD;
	__start__ --> route_question;
	route_question -.-> api_contract_graph\3acollect_contract_checks;
	route_question -.-> docs_graph\3akeyword_graph\3aextract_keywords;
	api_contract_graph\3asummarize_contract --> __end__;
	docs_graph\3asummarize_docs --> __end__;
	subgraph api_contract_graph
	api_contract_graph\3acollect_contract_checks --> api_contract_graph\3asummarize_contract;
	end
	subgraph docs_graph
	docs_graph\3akeyword_graph\3aprepare_keyword_context --> docs_graph\3asummarize_docs;
	subgraph keyword_graph
	docs_graph\3akeyword_graph\3aextract_keywords --> docs_graph\3akeyword_graph\3aprepare_keyword_context;
	end
	end
```

## Request Flow

```mermaid
flowchart TD
  request([OpenAI user message]) --> adapter["request_to_input"]
  adapter --> route["Normalize question and select route"]

  subgraph api["API contract subgraph"]
    checks["Collect contract checks"] --> api_summary["Stream contract summary"]
  end

  subgraph docs["Docs specialist subgraph"]
    subgraph keywords["Keyword grandchild subgraph"]
      extract["Extract keywords"] --> prepare["Build context and emit status"]
    end
    prepare --> docs_summary["Stream docs summary"]
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
non-streaming requests. Rich status delivery also requires a streaming request
with `metadata.langgraph_stream_events="v1"`.

The graph is deterministic and has no checkpointer or Store. Its routing,
keywords, checks, and messages exist only for the current request. Any
conversation history must come from messages supplied by the client.

## Try It

| Route | Prompt |
| --- | --- |
| API contract | `Show OpenAI adapter streaming with nested subgraphs.` |
| Documentation | `Show nested subgraph routing docs.` |
