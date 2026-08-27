# LGOS RAG

`lgos-rag` is an agentic retrieval graph over the Markdown corpus packaged with
the demo API. It lazily splits and embeds that corpus into a process-local
in-memory vector index, retrieves up to four chunks, and grounds answers with
the source URLs stored on those chunks.

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
streamed nodes. The index is a demo optimization, not durable application data.
