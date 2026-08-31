import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from dotenv import load_dotenv

    load_dotenv()

    API_PREFIX = "v1"
    BASE_URL = f"http://localhost:8000/{API_PREFIX}"
    return (BASE_URL,)


@app.cell
def _():
    import os

    api_key = os.getenv("OPENAI_API_KEY", None)

    if api_key is None:
        msg = "Please set the OPENAI_API_KEY environment variable."
        raise ValueError(msg)
    return (api_key,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Official Openai
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sync
    """)
    return


@app.cell
def _(BASE_URL, api_key):
    from openai import OpenAI

    _openai_client = OpenAI(base_url=BASE_URL, api_key=api_key)
    _chat_completion = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Mix and use your two tools and give me a result",
            }
        ],
        stream=False,
    )
    print(_chat_completion.choices[0].message.content)
    return (OpenAI,)


@app.cell
def _(BASE_URL, OpenAI, api_key):
    _openai_client = OpenAI(base_url=BASE_URL, api_key=api_key)
    _chat_completion = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Mix and use your two tools and give me a result",
            }
        ],
        stream=True,
    )
    for chunk in _chat_completion:
        print(chunk.choices[0].delta.content)
        print("****************")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Async
    """)
    return


@app.cell
async def _(BASE_URL, api_key):
    from openai import AsyncClient

    _async_client = AsyncClient(base_url=BASE_URL, api_key=api_key)
    async_chat_completion = await _async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Mix and use your two tools and give me a result",
            }
        ],
        stream=False,
    )
    print(async_chat_completion.choices[0].message.content)
    return (AsyncClient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Async Simultaneous Calls
    """)
    return


@app.cell
async def _(AsyncClient, BASE_URL, api_key):
    import asyncio

    async def get_completion(client: AsyncClient, prompt: str):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content

    prompts = [
        "Write a haiku about programming",
        "Explain quantum computing in one sentence",
        "Give me a recipe for chocolate chip cookies",
        "What are the three laws of robotics?",
    ]
    _async_client = AsyncClient(base_url=BASE_URL, api_key=api_key)

    async def run_concurrent_requests():
        tasks = [get_completion(_async_client, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    results = await run_concurrent_requests()
    for prompt, result in zip(prompts, results, strict=False):
        print(f"Prompt: {prompt}\n")
        print(f"Response: {result}\n")
        print("-" * 80)
    return


if __name__ == "__main__":
    app.run()
