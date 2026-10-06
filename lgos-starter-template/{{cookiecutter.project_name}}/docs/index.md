# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## First request

Run `just setup`, set `APP_OPENAI_API_KEY` in `.env`, and start the server with
`just run`. Development needs no database or other service: interrupt state
and background runs stay in the server process.

=== "Responses"

    ```bash
    uv run --locked --env-file .env examples/responses.py
    ```

=== "Streaming Responses"

    ```bash
    uv run --locked --env-file .env examples/responses.py --stream
    ```

=== "Chat Completions"

    ```bash
    uv run --locked --env-file .env examples/chat.py
    uv run --locked --env-file .env examples/chat.py --stream
    ```

The registered graph ID is the OpenAI `model` value. `APP_OPENAI_MODEL` selects
the upstream model invoked *inside* that graph. Clients send their conversation
history with each request; the service does not own a conversation store.

## Next steps

- [Add or replace a graph](graphs.md).
- [Configure the graphs and the server](configuration.md).
- [Run containers, PostgreSQL, Hatchet, and telemetry](deployment.md).
- Read the [LGOS server guide](https://ilkersigirci.github.io/langgraph-openai-serve/latest/how-to-guides/server/)
  for what `lgos serve` provides.
