from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_sec: int = 120
    temperature: float = 0.2

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
            "options": {"temperature": self.temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
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
        return content.strip()

