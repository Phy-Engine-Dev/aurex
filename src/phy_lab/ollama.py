from __future__ import annotations

import json
import threading
import urllib.parse
from queue import Queue
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    pass


def _is_local_base_url(url: str) -> bool:
    try:
        p = urllib.parse.urlparse((url or "").strip())
    except Exception:
        return False
    host = (p.hostname or "").strip().casefold()
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_sec: int = 120
    temperature: float = 0.2
    num_predict: int = 2048

    def chat(self, *, messages: list[dict[str, str]]) -> str:
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise OllamaError(
                "Missing dependency: requests (required to call the Ollama HTTP API)"
            ) from e

        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": int(self.num_predict)},
        }

        try:
            # Avoid accidentally proxying localhost (common when users set HTTP_PROXY for web search).
            session = requests.Session()
            if _is_local_base_url(self.base_url):
                session.trust_env = False
            resp = session.post(url, json=payload, timeout=self.timeout_sec)
        except requests.RequestException as e:
            raise OllamaError(f"Failed to reach Ollama at {url}: {e}") from e

        if not resp.ok:
            body = resp.text
            if len(body) > 2000:
                body = body[:2000] + "...(truncated)"
            raise OllamaError(f"Ollama error {resp.status_code}: {body}")

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise OllamaError(f"Invalid JSON response from Ollama: {e}") from e

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response missing 'message' object")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama response missing 'message.content' string")
        content = content.strip()
        if not content:
            raise OllamaError("Ollama returned empty 'message.content'")
        return content


class OllamaPool:
    """A simple client pool to allow concurrent Ollama requests.

    This is useful when running multiple Ollama servers (e.g. one per GPU) or when you
    want to increase throughput on a single server.
    """

    def __init__(self, clients: list[OllamaClient]):
        if not clients:
            raise OllamaError("OllamaPool requires at least one client")
        self._q: "Queue[OllamaClient]" = Queue()
        for c in clients:
            self._q.put(c)
        self._lock = threading.Lock()
        self._size = len(clients)

    @property
    def size(self) -> int:
        return self._size

    def chat(self, *, messages: list[dict[str, str]]) -> str:
        client = self._q.get()
        try:
            return client.chat(messages=messages)
        finally:
            self._q.put(client)
