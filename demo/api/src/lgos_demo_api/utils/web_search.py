"""Typed access to a self-hosted web-search JSON endpoint."""

import httpx
from pydantic import AnyHttpUrl, BaseModel, Field, OnErrorOmit

_RESULT_LIMIT = 5


class WebSearchResult(BaseModel):
    """Validated fields consumed by the demo web-search tool."""

    url: AnyHttpUrl
    title: str = ""
    content: str = ""


class _SearchResponse(BaseModel):
    results: list[OnErrorOmit[WebSearchResult]] = Field(default_factory=list)


async def search_web(
    client: httpx.AsyncClient,
    url: str,
    query: str,
) -> list[WebSearchResult]:
    """Fetch, validate, and deduplicate compatible JSON search results."""
    response = await client.get(url, params={"q": query, "format": "json"})
    response.raise_for_status()
    payload = _SearchResponse.model_validate(response.json())

    results: list[WebSearchResult] = []
    seen_urls: set[str] = set()
    for result in payload.results:
        url = str(result.url)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(result)
        if len(results) == _RESULT_LIMIT:
            break
    return results


__all__ = ["WebSearchResult", "search_web"]
