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
    gptoss_optimization: bool = False

    def _apply_gptoss_optimization(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not self.gptoss_optimization:
            return messages
        model_low = (self.model or "").casefold()
        if "gpt-oss" not in model_low:
            return messages
        # Avoid duplicating the header if the caller already inserted it.
        if messages:
            m0 = messages[0]
            if isinstance(m0, dict) and m0.get("role") == "system":
                c0 = str(m0.get("content") or "")
                if "reasoning: high" in c0 and "harmony" in c0.casefold():
                    return messages
        header = (
            "Harmony\n"
            "reasoning: high\n"
            "\n"
            "Guidance:\n"
            "- Do thorough internal reasoning.\n"
            "- Do NOT reveal chain-of-thought.\n"
            "- Follow all formatting/output instructions in later system messages.\n"
        )
        return [{"role": "system", "content": header}] + list(messages)

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
            "messages": self._apply_gptoss_optimization(list(messages or [])),
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": int(self.num_predict)},
        }

        # Avoid accidentally proxying localhost (common when users set HTTP_PROXY for web search).
        session = requests.Session()
        if _is_local_base_url(self.base_url):
            session.trust_env = False

        # Some Ollama builds/models can occasionally return an empty message.content.
        # Treat it as a transient server-side failure and retry once.
        max_attempts = 2
        last_data: Any = None
        for attempt in range(1, max_attempts + 1):
            try:
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

            last_data = data
            message = data.get("message")
            if not isinstance(message, dict):
                raise OllamaError("Ollama response missing 'message' object")
            content = message.get("content")
            if not isinstance(content, str):
                raise OllamaError("Ollama response missing 'message.content' string")
            content = content.strip()
            if content:
                return content
            if attempt < max_attempts:
                continue

        # If we get here, every attempt returned empty content.
        preview = ""
        try:
            preview = json.dumps(last_data, ensure_ascii=False)[:500]
        except Exception:
            preview = str(last_data)[:500]
        raise OllamaError(
            "Ollama returned empty 'message.content' (after retry). "
            + (f"Response preview: {preview}" if preview else "")
        )


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
