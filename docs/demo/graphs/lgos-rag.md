# LGOS RAG

`lgos-rag` is an agentic retrieval graph over the Markdown corpus packaged with
the demo API. It lazily splits and embeds that corpus into a process-local
in-memory vector index, retrieves up to four chunks, and grounds answers with
the source URLs stored on those chunks.

## LangGraph Topology

```mermaid
graph TD;
	__start__ --> generate_query_or_respond;
	generate_query_or_respond -.-> __end__;
	generate_query_or_respond -. &nbsp;tools&nbsp; .-> retrieve;
	retrieve -.-> answer_no_results;
	retrieve -.-> generate_answer;
	retrieve -.-> rewrite_question;
	rewrite_question --> generate_query_or_respond;
	answer_no_results --> __end__;
	generate_answer --> __end__;
```

## Request Flow

```mermaid
flowchart TD
  start([User message]) --> decide["Choose direct response or retrieval"]
  decide -->|"greeting, conversation, or unrelated"| direct["Direct response"]
  direct --> done([End])

  decide -->|"LGOS factual question"| retrieve["Retrieve documentation"]
  retrieve --> grade{"Context relevant?"}
  grade -->|"yes"| answer["Generate grounded answer with Markdown links"]
  answer --> done
  grade -->|"no, first miss"| rewrite["Rewrite query once"]
  rewrite --> decide
  grade -->|"no after rewrite"| no_results["Answer that documentation is insufficient"]
  no_results --> done
```

The retry is deliberately bounded to one rewrite. Routing, grading, and
rewriting use non-streaming internal model calls; retrieval uses the in-memory
vector index. Direct, grounded, and no-result answers are the user-visible
streamed nodes.

## State And Lifetime

The graph has no checkpointer or LangGraph Store. Any conversation history
comes from messages supplied by the client. Each API process builds its own
vector index lazily on the first retrieval, reuses it for that process lifetime,
and rebuilds it after a restart. This is a demo optimization, not durable
application data.

## Try It

| Path | Prompt |
| --- | --- |
| Direct response | `Who are you?` |
| Retrieval | `How do I configure LGOS runtime settings?` |
