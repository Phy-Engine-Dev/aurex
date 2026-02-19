from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from lxml.html import document_fromstring

from .registry import ToolError, ToolRuntime


_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _collapse_ws(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def _unwrap_ddg_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    try:
        pu = urlparse(u)
    except Exception:
        return u
    if (pu.netloc or "").endswith("duckduckgo.com") and pu.path.startswith("/l/"):
        qs = parse_qs(pu.query)
        uddg = qs.get("uddg", [""])[0]
        if isinstance(uddg, str) and uddg.strip():
            try:
                return unquote(uddg.strip())
            except Exception:
                return uddg.strip()
    return u


def _parse_duckduckgo_html(*, html: str, max_results: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        root = document_fromstring(html or "")
    except Exception:
        return out

    bodies = root.xpath("//div[contains(@class,'result__body')]")
    if not isinstance(bodies, list):
        bodies = []

    for body in bodies:
        if not hasattr(body, "xpath"):
            continue
        a_nodes = body.xpath(".//a[contains(@class,'result__a')][1]")
        if not a_nodes:
            continue
        a0 = a_nodes[0]
        title = _collapse_ws(getattr(a0, "text_content", lambda: "")())
        href = _unwrap_ddg_url(str(getattr(a0, "get", lambda _k: "")("href") or "").strip())

        sn_nodes = body.xpath(
            ".//a[contains(@class,'result__snippet')][1] | .//div[contains(@class,'result__snippet')][1] | .//span[contains(@class,'result__snippet')][1]"
        )
        snippet = ""
        if sn_nodes:
            snippet = _collapse_ws(getattr(sn_nodes[0], "text_content", lambda: "")())

        key = href or title
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "url": href, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


def _parse_bing_html(*, html: str, max_results: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        root = document_fromstring(html or "")
    except Exception:
        return out

    items = root.xpath("//li[contains(@class, 'b_algo')]")
    if not isinstance(items, list):
        items = []

    for li in items:
        if not hasattr(li, "xpath"):
            continue
        a_nodes = li.xpath(".//h2/a[1]")
        if not a_nodes:
            continue
        a0 = a_nodes[0]
        title = _collapse_ws(getattr(a0, "text_content", lambda: "")())
        href = str(getattr(a0, "get", lambda _k: "")("href") or "").strip()
        p_nodes = li.xpath(".//p[1]")
        snippet = ""
        if p_nodes:
            snippet = _collapse_ws(getattr(p_nodes[0], "text_content", lambda: "")())
        key = href or title
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "url": href, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


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
    if safesearch not in ("off", "moderate", "strict"):
        safesearch = "moderate"
    # DuckDuckGo HTML endpoint uses `kp` for safesearch, but we keep it best-effort.
    kp_map = {"off": "-2", "moderate": "-1", "strict": "1"}
    kp = kp_map.get(safesearch, "-1")

    data: dict[str, str] = {"q": query, "kp": kp}
    if region:
        data["kl"] = region
    tr = (time_range or "").strip().lower()
    if tr in ("d", "w", "m", "y"):
        data["df"] = tr

    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        "Origin": "https://duckduckgo.com",
        "Referer": "https://duckduckgo.com/",
    }

    try:
        resp = requests.post("https://html.duckduckgo.com/html/", data=data, headers=headers, timeout=20)
    except Exception as e:
        raise ToolError(f"web_search: request failed: {type(e).__name__}: {e}") from e

    if int(getattr(resp, "status_code", 0) or 0) != 200:
        raise ToolError(f"web_search: HTTP {getattr(resp, 'status_code', '?')}")

    html = str(getattr(resp, "text", "") or "")
    out = _parse_duckduckgo_html(html=html, max_results=max_results)

    # Fallback: Bing parsing via requests (best-effort).
    if not out:
        try:
            resp2 = requests.get(
                "https://www.bing.com/search",
                params={"q": query},
                headers={"User-Agent": _UA, "Accept-Language": headers["Accept-Language"]},
                timeout=20,
            )
        except Exception as e:
            raise ToolError(f"web_search: no results and bing fallback failed: {type(e).__name__}: {e}") from e
        if int(getattr(resp2, "status_code", 0) or 0) == 200:
            out = _parse_bing_html(html=str(getattr(resp2, "text", "") or ""), max_results=max_results)

    return out


WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Web search via DuckDuckGo HTML endpoint (best-effort; may fall back to Bing). Returns a small list of results.",
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
