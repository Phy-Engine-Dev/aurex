from __future__ import annotations

import json
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


class OllamaError(RuntimeError):
    pass


def _is_local_base_url(base_url: str) -> bool:
    try:
        u = urllib.parse.urlparse((base_url or "").strip())
    except Exception:
        return False
    host = (u.hostname or "").strip().casefold()
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def _join_url(base_url: str, path: str) -> str:
    base = (base_url or "").rstrip("/")
    p = "/" + (path or "").lstrip("/")
    return base + p


@dataclass(frozen=True)
class OllamaChatResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_sec: int = 240
    temperature: float = 0.2
    num_predict: int = 2048
    extra_options: dict[str, Any] | None = None

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: str | None = None,
    ) -> OllamaChatResponse:
        url = _join_url(self.base_url, "/api/chat")

        options: dict[str, Any] = {
            "temperature": float(self.temperature),
            "num_predict": int(self.num_predict),
        }
        if self.extra_options:
            options.update(dict(self.extra_options))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["format"] = response_format

        data = _post_json(
            url=url,
            payload=payload,
            timeout_sec=self.timeout_sec,
            disable_env_proxy=_is_local_base_url(self.base_url),
        )

        msg = data.get("message")
        if not isinstance(msg, dict):
            raise OllamaError("Ollama response missing message")

        content = msg.get("content")
        if not isinstance(content, str):
            content = ""
        content = content.strip()

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_calls_out = [tc for tc in tool_calls if isinstance(tc, dict)]
        else:
            tool_calls_out = []

        return OllamaChatResponse(content=content, tool_calls=tool_calls_out, raw=data)


def _post_json(*, url: str, payload: dict[str, Any], timeout_sec: int, disable_env_proxy: bool) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    handlers: list[urllib.request.BaseHandler] = []
    if disable_env_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)

    try:
        with opener.open(req, timeout=timeout_sec) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        txt = ""
        try:
            txt = (e.read() or b"")[:2000].decode("utf-8", errors="replace")
        except Exception:
            txt = ""
        raise OllamaError(f"Ollama HTTP error {e.code}: {txt}".rstrip()) from e
    except (urllib.error.URLError, socket.timeout) as e:
        raise OllamaError(f"Failed to reach Ollama at {url}: {e}") from e

    try:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        preview = raw[:500].decode("utf-8", errors="replace")
        raise OllamaError(f"Invalid JSON from Ollama: {e}: {preview}") from e

    if not isinstance(obj, dict):
        raise OllamaError("Ollama returned non-object JSON")
    return obj


def tool_schema(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def message(role: str, content: str, **extra: Any) -> dict[str, Any]:
    m: dict[str, Any] = {"role": role, "content": content}
    for k, v in extra.items():
        m[k] = v
    return m


def flatten_text_blocks(items: Iterable[Any]) -> str:
    parts: list[str] = []
    for it in items:
        if isinstance(it, str):
            parts.append(it)
            continue
        if isinstance(it, dict):
            t = it.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join([p for p in parts if p.strip()]).strip()

