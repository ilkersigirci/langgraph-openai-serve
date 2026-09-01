"""Translate LGOS client events into Open WebUI events and embeds."""

import json
from html import escape
from typing import Any

from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from .contracts import LGOS_EXTENSION_KEY, ChartArtifact


def _client_event(chunk: ChatCompletionChunk) -> dict[str, Any] | None:
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None

    event = extension.get("event")
    return event if isinstance(event, dict) else None


def _client_event_data(
    event: dict[str, Any],
    event_type: str,
) -> dict[str, Any] | None:
    if event.get("type") != event_type:
        return None
    data = event.get("data")
    return data if isinstance(data, dict) else None


def _status_event(
    event: dict[str, Any],
) -> dict[str, Any] | None:
    data = _client_event_data(event, "status")
    if data is None:
        return None

    description = data.get("description")
    done = data.get("done", False)
    hidden = data.get("hidden", False)
    if (
        not isinstance(description, str)
        or not description
        or not isinstance(done, bool)
        or not isinstance(hidden, bool)
    ):
        return None

    return {
        "type": "status",
        "data": {
            "description": description,
            "done": done,
            "hidden": hidden,
        },
    }


def _chart_embed_event(event: dict[str, Any]) -> dict[str, Any] | None:
    data = _client_event_data(event, "artifact")
    if data is None:
        return None
    try:
        artifact = ChartArtifact.model_validate(data)
    except ValidationError:
        return None

    html = _chart_html(artifact)
    if html is None:
        return None
    return {"type": "embeds", "data": {"embeds": [html], "replace": True}}


def _chart_html(artifact: ChartArtifact) -> str | None:
    trace_type = "bar" if artifact.chart_type == "bar" else "scatter"
    data: list[dict[str, object]] = []
    for series in artifact.series:
        trace: dict[str, object] = {
            "type": trace_type,
            "name": series.name,
            "x": artifact.labels,
            "y": series.values,
            "showlegend": artifact.show_legend,
        }
        if artifact.chart_type == "line":
            trace["mode"] = "lines+markers"
        data.append(trace)
    try:
        figure = json.dumps(
            {
                "data": data,
                "layout": {
                    "title": {"text": artifact.title},
                    "xaxis": {"title": {"text": artifact.x_axis_title}},
                    "yaxis": {"title": {"text": artifact.y_axis_title}},
                    "showlegend": artifact.show_legend,
                },
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).replace("<", "\\u003c")
    except (TypeError, ValueError):
        return None

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-4.0.0.min.js" charset="utf-8"></script>
</head><body style="margin:0;padding:16px">
<h2>{escape(artifact.title)}</h2><p>{escape(artifact.summary)}</p>
<div id="plot"></div>
<script>
const figure = {figure};
function reportHeight() {{
  const height = document.documentElement.scrollHeight;
  parent.postMessage({{type: "iframe:height", height}}, "*");
}}
window.addEventListener("load", reportHeight);
new ResizeObserver(reportHeight).observe(document.body);
Plotly.newPlot("plot", figure.data, figure.layout, {{responsive: true}});
</script></body></html>"""
