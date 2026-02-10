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
    for pfx in ("tool.", "tools.", "tool:", "tool/"):
        if name.startswith(pfx):
            name = name[len(pfx) :].strip()
            break
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
            "reasoning: medium\n"
            "\n"
            "Guidance:\n"
            "- Keep internal reasoning brief.\n"
            "- Do NOT reveal chain-of-thought.\n"
            "- Always produce a non-empty final answer in message.content.\n"
            "- Follow all formatting/output instructions in later system messages.\n"
        )
        return [{"role": "system", "content": header}] + list(messages)

    def chat(self, *, messages: list[dict[str, str]], response_format: str | None = None) -> str:
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
        rf = str(response_format or "").strip().lower()
        if rf:
            # Ollama supports `format: "json"` to force strict JSON outputs.
            payload["format"] = rf

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

        # Some Ollama builds/models can occasionally return an empty message.content,
        # especially when the prompt is near the context window limit or the model
        # spends the whole budget in a hidden "thinking" channel.
        max_attempts = 3
        last_data: Any = None
        last_details: dict[str, Any] = {}
        base_options: dict[str, Any] = dict(payload.get("options") or {})

        def _prune_messages_for_retry(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
            # Keep a small, high-signal subset to reduce prompt size and improve the
            # chance of producing a non-empty message.content.
            if not isinstance(msgs, list) or not msgs:
                return []
            out: list[dict[str, str]] = []
            # Preserve the first system message (often contains format constraints).
            m0 = msgs[0] if isinstance(msgs[0], dict) else None
            if isinstance(m0, dict) and isinstance(m0.get("role"), str) and isinstance(m0.get("content"), str):
                out.append({"role": m0["role"], "content": m0["content"]})
            # Preserve the second system message if it exists (common: tool list).
            if len(msgs) > 1:
                m1 = msgs[1] if isinstance(msgs[1], dict) else None
                if (
                    isinstance(m1, dict)
                    and m1.get("role") == "system"
                    and isinstance(m1.get("content"), str)
                    and (m1 not in out)
                ):
                    out.append({"role": "system", "content": m1["content"]})
            # Keep the most recent tail.
            tail = [m for m in msgs[-12:] if isinstance(m, dict)]
            for m in tail:
                role = m.get("role")
                content = m.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    continue
                if any((x.get("role") == role and x.get("content") == content) for x in out):
                    continue
                if len(content) > 2200:
                    content = content[:2000] + f"...<truncated len={len(content)}>"
                out.append({"role": role, "content": content})
            return out

        retry_guard2 = (
            "IMPORTANT (empty output fix):\n"
            "- Your previous responses had empty message.content. This is NOT allowed.\n"
            "- You MUST put the final answer/tool-call JSON in message.content.\n"
            "- Keep the response short and direct.\n"
        )

        for attempt in range(1, max_attempts + 1):
            try:
                resp = session.post(url, json=payload, timeout=self.timeout_sec)
            except requests.RequestException as e:
                raise OllamaError(f"Failed to reach Ollama at {url}: {e}") from e

            if not resp.ok:
                body = resp.text
                body_low = (body or "").casefold()
                # Some older Ollama servers may not support the `format` field.
                if (
                    rf
                    and attempt == 1
                    and int(getattr(resp, "status_code", 0) or 0) in (400, 404)
                    and "format" in body_low
                    and any(x in body_low for x in ("unknown", "unsupported", "unrecognized", "invalid"))
                ):
                    payload.pop("format", None)
                    rf = ""
                    continue
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
                # Retry with additional guards + a smaller generation budget.
                retry_num_predict = int(base_options.get("num_predict") or self.num_predict)
                if retry_num_predict <= 0:
                    retry_num_predict = int(self.num_predict) if int(self.num_predict) > 0 else 512
                retry_num_predict = min(retry_num_predict, 512)
                payload["options"] = dict(base_options)
                payload["options"]["num_predict"] = retry_num_predict

                if attempt == 1:
                    payload["messages"] = list(base_messages) + [{"role": "system", "content": retry_guard}]
                else:
                    # Final retry: aggressively prune the prompt and re-guard.
                    payload["messages"] = _prune_messages_for_retry(list(base_messages)) + [
                        {"role": "system", "content": retry_guard2},
                        {"role": "system", "content": retry_guard},
                    ]
                continue

        # Last-chance fallback when strict format=json yields empty content.
        # Some thinking-heavy models occasionally return only a hidden "thinking" field
        # (content="") even after retries. Try once without `format`, while still
        # requiring JSON in message.content.
        if rf:
            try:
                payload2: dict[str, Any] = dict(payload)
                payload2.pop("format", None)
                payload2["options"] = dict(base_options)
                retry_num_predict = int(base_options.get("num_predict") or self.num_predict)
                if retry_num_predict <= 0:
                    retry_num_predict = int(self.num_predict) if int(self.num_predict) > 0 else 512
                payload2["options"]["num_predict"] = min(retry_num_predict, 512)
                payload2["messages"] = _prune_messages_for_retry(list(base_messages)) + [
                    {
                        "role": "system",
                        "content": (
                            "FINAL RETRY (format fallback):\n"
                            "- You MUST return a non-empty JSON object in message.content.\n"
                            "- Do NOT output only hidden thinking.\n"
                            "- If you intend a tool call, output the tool-call JSON object.\n"
                        ),
                    }
                ]
                resp2 = session.post(url, json=payload2, timeout=self.timeout_sec)
                if resp2.ok:
                    data2 = resp2.json()
                    last_data = data2
                    message2 = data2.get("message")
                    if isinstance(message2, dict):
                        content2 = message2.get("content")
                        if isinstance(content2, str) and content2.strip():
                            return content2.strip()
                        tool_json2 = _tool_calls_to_agent_json(message2.get("tool_calls"))
                        if isinstance(tool_json2, str) and tool_json2.strip():
                            return tool_json2
            except Exception:
                pass

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

    def chat(self, *, messages: list[dict[str, str]], response_format: str | None = None) -> str:
        client = self._q.get()
        try:
            return client.chat(messages=messages, response_format=response_format)
        finally:
            self._q.put(client)
