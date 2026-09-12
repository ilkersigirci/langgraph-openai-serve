"""Typed access to a self-hosted web-search JSON endpoint."""

from typing import Any

import httpx
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError

_RESULT_LIMIT = 5


class _SearchResponse(BaseModel):
    results: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class WebSearchResult(BaseModel):
    """Validated fields consumed by the demo web-search tool."""

    url: AnyHttpUrl
    title: str = ""
    content: str = ""

    model_config = ConfigDict(extra="ignore")


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
    for raw_result in payload.results:
        try:
            result = WebSearchResult.model_validate(raw_result)
        except ValidationError:
            continue
        url = str(result.url)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(result)
        if len(results) == _RESULT_LIMIT:
            break
    return results


__all__ = ["WebSearchResult", "search_web"]
