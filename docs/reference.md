# API Reference

This page provides detailed information about the LangGraph OpenAI Serve
OpenAI-compatible endpoints and schemas. The default API prefix is `/v1`; set
`LGOS_OPENAI_API_PREFIX` or pass `prefix=` to `bind_openai_chat_completion()` to
mount the endpoints elsewhere.

FastAPI docs for the mounted OpenAI API are disabled by default. Set
`LGOS_OPENAI_API_DOCS_ENABLED=true` to expose `{prefix}/docs`,
`{prefix}/redoc`, and `{prefix}/openapi.json`.

::: langgraph_openai_serve
