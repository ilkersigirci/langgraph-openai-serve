import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from openai import AsyncOpenAI, OpenAI

    BASE_URL = "http://localhost:3004/v1"
    MODEL = "custom-input-output-context"
    return AsyncOpenAI, BASE_URL, MODEL, OpenAI, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Call LGOS with the official OpenAI Responses client

    Start the demo with `make run-api-local` from `demo/`. These examples use
    the deterministic custom-schema graph, so no upstream model key is needed.
    Each call is stateless: send `store=False` and carry any history in `input`.
    """)
    return


@app.cell
def _(BASE_URL, MODEL, OpenAI):
    with OpenAI(base_url=BASE_URL, api_key="DUMMY") as _client:
        _response = _client.responses.create(
            model=MODEL,
            input="Show me custom schemas.",
            user="demo-user",
            store=False,
        )
        print(_response.output_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Streaming final answers
    """)
    return


@app.cell
def _(BASE_URL, MODEL, OpenAI):
    with (
        OpenAI(base_url=BASE_URL, api_key="DUMMY") as _client,
        _client.responses.stream(
            model=MODEL,
            input="Stream the custom-schema answer.",
            user="demo-user",
            store=False,
        ) as _stream,
    ):
        _phases = {}
        for _event in _stream:
            if (
                _event.type == "response.output_item.added"
                and _event.item.type == "message"
            ):
                _phases[_event.output_index] = _event.item.phase
            elif (
                _event.type == "response.output_text.delta"
                and _phases.get(_event.output_index) == "final_answer"
            ):
                print(_event.delta, end="", flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concurrent asynchronous requests
    """)
    return


@app.cell
async def _(AsyncOpenAI, BASE_URL, MODEL):
    import asyncio

    _prompts = [
        "Show me graph input.",
        "Show me graph context.",
        "Show me graph output.",
    ]
    async with AsyncOpenAI(base_url=BASE_URL, api_key="DUMMY") as _client:
        _responses = await asyncio.gather(
            *(
                _client.responses.create(
                    model=MODEL, input=prompt, user="demo-user", store=False
                )
                for prompt in _prompts
            )
        )
    for _prompt, _response in zip(_prompts, _responses, strict=True):
        print(f"Prompt: {_prompt}\nResponse: {_response.output_text}\n")
    return


if __name__ == "__main__":
    app.run()
