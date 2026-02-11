from __future__ import annotations

from typing import Any

from .registry import ToolError, ToolRuntime


def ddg_web_search(runtime: ToolRuntime, args: dict[str, Any]) -> list[dict[str, str]]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("web_search: query is empty")

    max_results = int(args.get("max_results") or 5)
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    region = str(args.get("region") or getattr(runtime.config.web_search, "region", "cn-zh"))
    safesearch = str(args.get("safesearch") or getattr(runtime.config.web_search, "safesearch", "moderate"))
    time_range = str(args.get("time_range") or getattr(runtime.config.web_search, "time_range", "y"))

    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ToolError(
            "Missing dependency: duckduckgo-search. Install it with `pip install duckduckgo-search`."
        ) from e

    out: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            region=region,
            safesearch=safesearch,
            timelimit=time_range,
            max_results=max_results,
        ):
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or "").strip()
            href = str(r.get("href") or r.get("url") or "").strip()
            body = str(r.get("body") or r.get("snippet") or "").strip()
            if not title and not href and not body:
                continue
            out.append({"title": title, "url": href, "snippet": body})

    return out


WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Web search via DuckDuckGo (duckduckgo-search library only). Returns a small list of results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "region": {"type": "string", "description": "DDG region, e.g. cn-zh, us-en."},
            "safesearch": {"type": "string", "enum": ["off", "moderate", "strict"], "default": "moderate"},
            "time_range": {
                "type": "string",
                "description": "DDG timelimit, e.g. d,w,m,y (day/week/month/year).",
                "default": "y",
            },
        },
        "required": ["query"],
    },
}

