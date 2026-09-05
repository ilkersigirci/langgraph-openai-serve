import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stream and non-stream graph parity

    Run the deterministic `citation-events` demo graph twice and verify that
    streaming produces the same assistant text as a regular invocation. First
    call the LGOS graph runner directly, then call the served API with the
    official OpenAI client.
    """)
    return


@app.cell
def _():
    from langgraph_openai_serve import GraphRegistry
    from langgraph_openai_serve.api.responses.request import decode_responses_request
    from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
    from langgraph_openai_serve.graph.runner import (
        run_langgraph,
        run_langgraph_stream,
    )

    from lgos_demo_api.graphs.citations import citation_graph_config

    MODEL = "citation-events"
    request = ResponseCreateRequest(
        model=MODEL,
        input="Show me a cited answer.",
        store=False,
    )
    request_options = request.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"stream"},
    )
    graph_registry = GraphRegistry(
        registry={MODEL: citation_graph_config},
    )

    def check_parity(path: str, complete: str, streamed: str):
        if complete != streamed:
            raise AssertionError(f"{path} stream and non-stream outputs differ")
        return {"path": path, "matches": True, "characters": len(complete)}

    return (
        MODEL,
        check_parity,
        decode_responses_request,
        graph_registry,
        request,
        request_options,
        run_langgraph,
        run_langgraph_stream,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Direct LGOS graph runner
    """)
    return


@app.cell
async def _(
    decode_responses_request,
    check_parity,
    graph_registry,
    request,
    run_langgraph,
    run_langgraph_stream,
):
    _request, _messages, _ = decode_responses_request(request)
    _complete = await run_langgraph(_request, _messages, graph_registry)
    _events = [
        event
        async for event in run_langgraph_stream(
            _request,
            _messages,
            graph_registry,
        )
    ]

    runner_text = str(_complete.output.text)
    _streamed_text = "".join(event for event in _events if isinstance(event, str))
    runner_result = check_parity("LGOS runner", runner_text, _streamed_text)
    return runner_result, runner_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Official OpenAI client

    Start the demo API with `make run-api-local` from `demo/`. It serves LGOS at
    `http://localhost:3004/v1`.
    """)
    return


@app.cell
async def _(check_parity, request_options):
    from openai import AsyncOpenAI

    async with AsyncOpenAI(
        base_url="http://localhost:3004/v1",
        api_key="DUMMY",
    ) as _client:
        _complete = await _client.responses.create(**request_options)
        _stream = await _client.responses.create(
            **request_options,
            stream=True,
        )
        _streamed_text = "".join(
            [
                event.delta
                async for event in _stream
                if event.type == "response.output_text.delta"
            ]
        )

    openai_text = _complete.output_text
    openai_result = check_parity("OpenAI client", openai_text, _streamed_text)
    return openai_result, openai_text


@app.cell(hide_code=True)
def _(mo, openai_result, openai_text, runner_result, runner_text):
    mo.vstack(
        [
            mo.md("## Result"),
            mo.ui.table(
                [runner_result, openai_result],
                pagination=False,
                selection=None,
            ),
            mo.accordion(
                {
                    "Direct runner answer": mo.md(runner_text),
                    "OpenAI client answer": mo.md(openai_text),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
