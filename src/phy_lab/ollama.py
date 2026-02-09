from __future__ import annotations

import json
import threading
import urllib.parse
from queue import Queue
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


def _safe_response_preview(data: Any, *, max_chars: int = 500) -> str:
    """Best-effort JSON preview that never includes chain-of-thought fields."""
    try:
        if isinstance(data, dict):
            d: dict[str, Any] = dict(data)
            msg = d.get("message")
            if isinstance(msg, dict):
                msg2: dict[str, Any] = dict(msg)
                # Some models include a 'thinking' field; never log it.
                if "thinking" in msg2:
                    msg2["thinking"] = "<omitted>"
                d["message"] = msg2
            return json.dumps(d, ensure_ascii=False)[:max_chars]
        return json.dumps(data, ensure_ascii=False)[:max_chars]
    except Exception:
        try:
            return (str(data) or "")[:max_chars]
        except Exception:
            return ""

def _tool_calls_to_agent_json(tool_calls: Any) -> str | None:
    """Convert Ollama/OpenAI-style tool_calls into our agent JSON tool-call format.

    Some models (notably gpt-oss) may return message.content="" but include tool calls in
    message.tool_calls. Our agent expects the tool call to be encoded in content.
    """
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    tc0 = tool_calls[0]
    if not isinstance(tc0, dict):
        return None
    func = tc0.get("function")
    if not isinstance(func, dict):
        return None
    name = func.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if name.startswith("tool_"):
        name = name[len("tool_") :].strip()

    # Known aliasing when models invent a namespace prefix.
    aliases = {
        "plar_list_plar": "list_plar",
        "plar_search_plar": "search_plar",
        "plar_web_search": "web_search",
        "google": "web_search",
    }
    tool = aliases.get(name, name)

    arguments = func.get("arguments")
    args_obj: dict[str, Any] = {}
    if isinstance(arguments, dict):
        args_obj = arguments
    elif isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                args_obj = parsed
        except Exception:
            args_obj = {}

    if tool == "end":
        final = ""
        for k in ("final", "answer", "content", "message", "text"):
            v = args_obj.get(k)
            if isinstance(v, str) and v.strip():
                final = v.strip()
                break
        return json.dumps({"tool": "end", "final": final}, ensure_ascii=False)

    return json.dumps({"tool": tool, "args": args_obj}, ensure_ascii=False)


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
        base_messages = self._apply_gptoss_optimization(list(messages or []))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": base_messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": int(self.num_predict)},
        }

        # Avoid accidentally proxying localhost (common when users set HTTP_PROXY for web search).
        session = requests.Session()
        if _is_local_base_url(self.base_url):
            session.trust_env = False

        retry_guard = (
            "IMPORTANT (non-empty output required):\n"
            "- Your previous response had empty 'message.content'. This is NOT allowed.\n"
            "- You MUST return a non-empty final answer in message.content.\n"
            "- Do NOT output only internal reasoning/thinking.\n"
            "- Keep the final answer short (<= 800 characters).\n"
            "- If you intend to call a tool, also include the tool-call JSON in message.content.\n"
        )

        # Some Ollama builds/models can occasionally return an empty message.content.
        # Treat it as a transient server-side failure and retry once.
        max_attempts = 2
        last_data: Any = None
        last_details: dict[str, Any] = {}
        base_options: dict[str, Any] = dict(payload.get("options") or {})
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
            # Some models return tool calls with empty content.
            tool_json = _tool_calls_to_agent_json(message.get("tool_calls"))
            if isinstance(tool_json, str) and tool_json.strip():
                return tool_json
            thinking = message.get("thinking")
            thinking_len = len(thinking) if isinstance(thinking, str) else 0
            done_reason = data.get("done_reason")
            eval_count = data.get("eval_count")
            prompt_eval_count = data.get("prompt_eval_count")
            last_details = {
                "model": self.model,
                "base_url": self.base_url,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "content_len": len(content),
                "had_thinking": bool(thinking_len),
                "thinking_len": thinking_len,
                "done_reason": done_reason,
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "num_predict": int(base_options.get("num_predict") or self.num_predict),
            }
            if attempt < max_attempts:
                # Retry with an extra guard message to encourage a non-empty final output.
                # If the model is burning the whole budget on the thinking channel, force a
                # shorter retry to give it a chance to produce message.content.
                retry_num_predict = int(base_options.get("num_predict") or self.num_predict)
                if retry_num_predict <= 0:
                    retry_num_predict = int(self.num_predict) if int(self.num_predict) > 0 else 512
                retry_num_predict = min(retry_num_predict, 512)
                payload["options"] = dict(base_options)
                payload["options"]["num_predict"] = retry_num_predict
                payload["messages"] = list(base_messages) + [{"role": "system", "content": retry_guard}]
                continue

        # If we get here, every attempt returned empty content.
        preview = _safe_response_preview(last_data, max_chars=500)
        detail_bits = []
        if last_details:
            if last_details.get("had_thinking"):
                detail_bits.append(f"thinking_len={int(last_details.get('thinking_len') or 0)}")
            detail_bits.append(f"attempts={int(last_details.get('max_attempts') or max_attempts)}")
        details_str = (" (" + ", ".join(detail_bits) + ")") if detail_bits else ""
        raise OllamaError(
            "Ollama returned empty 'message.content' (after retry)"
            + details_str
            + (f". Response preview: {preview}" if preview else "."),
            details=last_details or None,
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
