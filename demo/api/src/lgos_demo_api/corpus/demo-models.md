# Bundled demo models

The demo API registers these graph names as OpenAI models:

- `simple-graph` demonstrates conversation-history and audience runtime
  settings.
- `citation-events` emits structured OpenAI URL citations alongside portable
  Markdown links.
- `lgos-rag` retrieves from this packaged Markdown corpus, grades relevance,
  performs at most one query rewrite, and grounds its answer in retrieved text.
- `custom-input-output-context` demonstrates graph input, output, and context
  adapters.
- `advanced-mcp-tools` demonstrates an asynchronous graph factory and mock
  MCP-style tools.
- `complex-subgraphs` demonstrates routing across nested specialist graphs.
- `custom-event-showcase` streams explicitly public status, progress, and
  artifact events among ordinary assistant text.
- `interruptible-approval` persists an approval interrupt and resumes it from a
  standard OpenAI tool result.

`GET /v1/models` lists the registered names. Detailed retrieval of one model
can additionally advertise versioned LGOS features and safe client settings.
The [demo graph catalog](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/graphs.md)
summarizes the models and their runtime requirements.
