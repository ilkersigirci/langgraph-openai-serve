import httpx

from lgos_demo_api.utils.web_search import search_web


async def test_search_uses_json_and_filters_bad_results() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/search"
        assert dict(request.url.params) == {"q": "LangGraph", "format": "json"}
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "LangGraph",
                        "url": "https://docs.langchain.com/oss/python/langgraph/",
                        "content": "Build stateful agents.",
                    },
                    {
                        "title": "Duplicate",
                        "url": "https://docs.langchain.com/oss/python/langgraph/",
                    },
                    {"title": "Invalid", "url": "file:///etc/passwd"},
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
    ) as client:
        results = await search_web(
            client,
            "https://search.example/api/search",
            "LangGraph",
        )

    assert len(results) == 1
    assert results[0].title == "LangGraph"
    assert str(results[0].url) == "https://docs.langchain.com/oss/python/langgraph/"
