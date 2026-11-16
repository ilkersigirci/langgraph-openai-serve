"""Show the files a graph generates in Open WebUI."""

import httpx
from openai.types.responses import ResponseFunctionToolCall

from .contracts import (
    PLOTLY_MEDIA_TYPE,
    DisplayFileArguments,
    OpenWebUIEventEmitter,
    OpenWebUIRequest,
    PlotlyFigure,
)
from .gateway import _client


def _plotly_html(content: bytes) -> str:
    figure = PlotlyFigure.model_validate_json(content).model_dump_json(
        exclude_unset=True
    )
    # JSON is embedded inside a script: prevent labels from closing that element.
    figure = figure.replace("<", "\\u003c")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-4.0.0.min.js" charset="utf-8"></script>
</head><body style="margin:0">
<div id="plot" style="height:450px"></div>
<script>
const figure = {figure};
Plotly.newPlot("plot", {{...figure, config: {{responsive: true}}}}).then(plot => {{
  parent.postMessage({{type: "iframe:height", height: plot.offsetHeight}}, "*");
}});
</script></body></html>"""


async def _handle_display_file(
    call: ResponseFunctionToolCall,
    event_emitter: OpenWebUIEventEmitter | None,
    request: OpenWebUIRequest | None,
    *,
    files_base_url: str,
    api_key: str,
    timeout: float,  # ruff: ignore[async-function-with-timeout] - Forward the HTTP client timeout; this is not a task cancellation scope.
    provider: str,
) -> dict[str, str]:
    """Persist a generated image or interactive chart through native UI events."""
    if event_emitter is None:
        msg = "Open WebUI did not provide an event emitter for display_file."
        raise ValueError(msg)
    try:
        arguments = DisplayFileArguments.model_validate_json(call.arguments)
    except ValueError as exc:
        msg = "The display_file call contains invalid arguments."
        raise ValueError(msg) from exc

    async with _client(
        base_url=files_base_url,
        api_key=api_key,
        timeout=timeout,
    ) as client:
        download = await client.files.content(
            arguments.file_id,
            extra_query={"provider": provider},
        )
        content = await download.aread()

    if arguments.media_type == PLOTLY_MEDIA_TYPE:
        event = {"type": "embeds", "data": {"embeds": [_plotly_html(content)]}}
    else:
        async with _openwebui_client(request, timeout) as openwebui:
            response = await openwebui.post(
                "/api/v1/files/",
                params={"process": "false"},
                files={"file": (arguments.filename, content, arguments.media_type)},
            )
            response.raise_for_status()
        event = {
            "type": "files",
            "data": {
                "files": [
                    {
                        "type": "image",
                        "url": f"/api/v1/files/{response.json()['id']}/content",
                        "name": arguments.filename,
                    }
                ]
            },
        }
    await event_emitter(event)
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }


def _openwebui_client(
    request: OpenWebUIRequest | None,
    timeout: float,
) -> httpx.AsyncClient:
    """Call the running Open WebUI app in-process with the caller's credentials."""
    authorization = request.headers.get("authorization") if request else None
    if request is None or not authorization:
        msg = "Open WebUI request credentials are unavailable for file transfer."
        raise ValueError(msg)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=request.app),
        base_url="http://openwebui.internal",
        headers={"Authorization": authorization},
        timeout=timeout,
    )
