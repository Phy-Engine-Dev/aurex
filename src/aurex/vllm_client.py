"""OpenAI-compatible multimodal streaming transport, with explicit thinking."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import httpx
import requests


# ``None`` is a meaningful OpenAI request choice: omit ``max_tokens`` and let
# the server use the context window that remains after the prompt.  Keep a
# private sentinel for callers that merely omit the argument and therefore want
# the configured default.  Without this distinction, the agent's explicit
# ``max_tokens=None`` on its first thinking turn silently fell back to the
# historical 4096-token configuration limit.
_DEFAULT_MAX_TOKENS = object()


class ModelError(RuntimeError):
    pass


@dataclass
class ModelReply:
    content: str
    reasoning: str
    tool_calls: list[dict]
    usage: dict
    finish_reason: str
    # Safe per-response metadata only.  Partial content/tool arguments stay in
    # the fields above for private archival and are never copied here.
    generation_progress: dict | None = None


class InvalidToolCall(ModelError):
    """A completed response has an invalid, entirely unexecuted tool batch.

    Keep the assembled response for private archival/review. The diagnostic is
    ordinary application text and must not embed model arguments or reasoning.
    Transport loss, cancellation and generation limits use their existing paths.
    """

    def __init__(self, diagnostic: str, reply: ModelReply):
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        self.reply = reply


class DegenerateGeneration(ModelError):
    """One streamed response became mechanically repetitive or hit its bound.

    This aborts only the current generation and never executes its partial tool
    call. The agent can re-orient from durable task/workspace state on its next
    no-thinking turn.
    """

    def __init__(self, diagnostic: str, reply: ModelReply, progress: dict):
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        self.reply = reply
        self.progress = progress


class _TokenProgress:
    """Bounded, non-decoding diagnostics with an extreme per-response fuse.

    Only the last 64 generated IDs are retained in memory for exact comparisons.
    Neither those IDs nor prompt IDs are emitted or persisted. This is not a
    semantic loop detector: legitimate arrays can contain exact repeated tokens.
    """
    PERIOD_LIMIT = 64
    INTERVAL_SECONDS = 5.0
    SAME_TOKEN_FUSE = 256
    SHORT_PERIOD_FUSE = 1024
    SHORT_PERIOD_MAX = 16

    def __init__(self):
        self.count = 0
        self._hash = hashlib.sha256()
        self._history = [None] * self.PERIOD_LIMIT
        self._matches = [0] * (self.PERIOD_LIMIT + 1)
        self._contiguous = 0
        self.invalid_entries = 0
        self.max_same_token_run = 0
        self.max_repeat = {'period_tokens': 0, 'copies': 0, 'span_tokens': 0}
        self.started = time.monotonic()
        self.last_emit = self.started
        self.last_emitted_count = 0

    def observe(self, token_ids) -> None:
        # Unknown/malformed diagnostic fields must not break a valid completion.
        if not isinstance(token_ids, list):
            if token_ids is not None:
                self._gap()
            return
        for token in token_ids:
            if type(token) is not int or not 0 <= token <= 0xffffffff:
                self._gap()
                continue
            for period in range(1, min(self.PERIOD_LIMIT, self._contiguous) + 1):
                previous = self._history[(self.count - period) % self.PERIOD_LIMIT]
                self._matches[period] = self._matches[period] + 1 if token == previous else 0
                copies = self._matches[period] // period + 1
                span = copies * period if copies >= 2 else 0
                if (span > self.max_repeat['span_tokens'] or
                        (span and span == self.max_repeat['span_tokens'] and
                         period < self.max_repeat['period_tokens'])):
                    self.max_repeat = {'period_tokens': period, 'copies': copies, 'span_tokens': span}
            self.max_same_token_run = max(self.max_same_token_run, self._matches[1] + 1)
            self._history[self.count % self.PERIOD_LIMIT] = token
            self._hash.update(token.to_bytes(4, byteorder='big'))
            self.count += 1
            self._contiguous += 1

    def _gap(self):
        # Do not claim exact adjacency across a malformed/missing diagnostic fragment.
        self.invalid_entries += 1
        self._contiguous = 0
        self._matches = [0] * (self.PERIOD_LIMIT + 1)

    def take(self, *, final: bool = False) -> str | None:
        now = time.monotonic()
        if self.count == self.last_emitted_count or (not final and now - self.last_emit < self.INTERVAL_SECONDS):
            return None
        self.last_emit, self.last_emitted_count = now, self.count
        return json.dumps({
            'schema': 'aurex.generated-token-progress.v1',
            'generated_tokens': self.count,
            'generated_sha256': self._hash.hexdigest(),
            'max_same_token_run': self.max_same_token_run,
            'max_exact_repeat': dict(self.max_repeat),
            'observed_period_limit': self.PERIOD_LIMIT,
            'invalid_token_entries': self.invalid_entries,
            'elapsed_s': round(max(0.0, now - self.started), 3),
        }, separators=(',', ':'))

    def degenerate_reason(self) -> str | None:
        """Return a bounded reason only for unmistakable mechanical output."""
        if self.max_same_token_run >= self.SAME_TOKEN_FUSE:
            return 'same_token_run'
        repeat = self.max_repeat
        if (0 < repeat['period_tokens'] <= self.SHORT_PERIOD_MAX and
                repeat['span_tokens'] >= self.SHORT_PERIOD_FUSE):
            return 'short_period_cycle'
        return None


class VLLMClient:
    # These are transport liveness/cancellation intervals, not generation budgets.
    _TICK_INTERVAL = 0.5
    _HEALTH_INTERVAL = 30.0
    _HEALTH_TIMEOUT = 5.0
    _HEALTH_FAILURES = 3
    _CLEANUP_TIMEOUT = 2.0

    def __init__(self, config):
        self.config = config
        self.base = config.base_url.rstrip('/')
        self.http = requests.Session()
        self.http.trust_env = False
        self._count_cache = OrderedDict()
        self._count_lock = threading.RLock()
        key = os.environ.get(config.api_key_env, '')
        if key:
            self.http.headers['Authorization'] = 'Bearer ' + key

    def capacity(self) -> int:
        # A task/model capacity refresh also invalidates tokenizer identity.
        with self._count_lock:
            self._count_cache.clear()
        r = self.http.get(self.base + '/models', timeout=10)
        r.raise_for_status()
        for model in r.json().get('data', []):
            if model.get('id') == self.config.model:
                return min(self.config.context_length, int(model.get('max_model_len') or self.config.context_length))
        raise ModelError(f'Model {self.config.model} is not served by {self.base}')

    def output_reserve_tokens(self, capacity: int) -> int:
        """Input-side headroom, not a task budget or a generated-token limit.

        OpenAI-compatible model listings often publish only the total window.
        Without an explicit configured limit, retain a fraction of that window
        for output. The agent may intentionally pass explicit None on the
        first full-context/final-answer response while separately bounding
        post-first tool navigation; direct callers retain the same distinction.
        """
        configured = self.config.max_output_tokens
        return int(configured) if configured is not None else max(1, capacity // 4)

    def _template_kwargs(self, thinking: bool) -> dict:
        options = {'enable_thinking': thinking}
        effort = getattr(self.config, 'reasoning_effort', None)
        if effort is not None:
            options['reasoning_effort'] = effort
        return options

    def count(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Prefer the server tokenizer, with a conservative UTF-8 fallback."""
        root = self.base[:-3] if self.base.endswith('/v1') else self.base
        payload = {'model': self.config.model, 'messages': messages,
                   'chat_template_kwargs': self._template_kwargs(self.config.enable_thinking),
                   'add_generation_prompt': True}
        if tools:
            payload['tools'] = tools
        # Retain only digest/count metadata, never prompts or image data. Key
        # order is intentionally preserved: template serialization may depend
        # on it. External image URLs can change contents without changing URL.
        cacheable = not any(
            p.get('type') == 'image_url' and not str((p.get('image_url') or {}).get('url', '')).startswith('data:')
            for m in messages for p in (m.get('content') if isinstance(m.get('content'), list) else [])
            if isinstance(p, dict))
        key = hashlib.sha256((root + json.dumps(payload, ensure_ascii=False)).encode()).hexdigest()
        now = time.monotonic()
        if cacheable:
            with self._count_lock:
                cached = self._count_cache.get(key)
                if cached and now - cached[0] < 30:
                    self._count_cache.move_to_end(key)
                    return cached[1]
        try:
            # Multimodal tokenization may load images; let the server count real visual tokens.
            r = self.http.post(root + '/tokenize', json=payload, timeout=30)
            count = r.json().get('count') if r.ok else None
            if type(count) is int and count >= 0:
                if cacheable:
                    with self._count_lock:
                        self._count_cache[key] = (time.monotonic(), count)
                        self._count_cache.move_to_end(key)
                        while len(self._count_cache) > 256:
                            self._count_cache.popitem(last=False)
                return count
        except (requests.RequestException, ValueError):
            pass
        total = 512
        for m in messages:
            content = m.get('content', '')
            if isinstance(content, list):
                total += sum(4096 if x.get('type') == 'image_url' else len(str(x.get('text', '')).encode('utf-8')) for x in content)
            else:
                total += len(str(content).encode('utf-8'))
            total += len(json.dumps(m.get('tool_calls', []), ensure_ascii=False).encode('utf-8')) + 32
        return total + len(json.dumps(tools or [], ensure_ascii=False).encode('utf-8'))

    def _stream_lines(self, payload: dict,
                      on_tick: Callable[[], None] | None = None) -> Iterator[bytes]:
        """Bridge cancellable asynchronous I/O into the synchronous agent loop.

        A tool parser may withhold SSE for minutes while tokens are still being
        generated. Silence is not a model timeout. During silence, use a separate
        read-only /models probe to detect sustained service loss, and keep the
        caller's cancellation callback running. A healthy but stalled engine is
        deliberately not declared failed from silence alone; it remains cancellable.
        Neither this bridge nor the probes retry a generation request.
        """
        events: queue.Queue = queue.Queue()
        stopped = threading.Event()
        state: dict[str, Any] = {}

        async def receive() -> None:
            state['loop'] = asyncio.get_running_loop()
            state['task'] = asyncio.current_task()
            if stopped.is_set():
                return
            timeout = httpx.Timeout(connect=10.0, write=30.0, pool=10.0, read=None)
            async with httpx.AsyncClient(headers=dict(self.http.headers),
                                         trust_env=False, timeout=timeout) as client:
                last_line = time.monotonic()

                async def read() -> None:
                    nonlocal last_line
                    async with client.stream('POST', self.base + '/chat/completions',
                                             json=payload) as response:
                        if not response.is_success:
                            # Bound error-body reads too: never hang while reporting a failure.
                            try:
                                body = await asyncio.wait_for(response.aread(), self._HEALTH_TIMEOUT)
                            except (asyncio.TimeoutError, httpx.HTTPError):
                                body = b'(error body unavailable)'
                            raise ModelError(f'vLLM HTTP {response.status_code}: {body[:1500].decode(errors="replace")}')
                        async for line in response.aiter_lines():
                            last_line = time.monotonic()
                            events.put(('line', line.encode('utf-8')))

                reader = asyncio.create_task(read())
                failures = 0
                try:
                    while not reader.done():
                        done, _ = await asyncio.wait({reader}, timeout=self._HEALTH_INTERVAL)
                        if done:
                            break
                        if time.monotonic() - last_line < self._HEALTH_INTERVAL:
                            failures = 0
                            continue
                        before_probe = last_line
                        try:
                            health = await asyncio.wait_for(
                                client.get(self.base + '/models', timeout=self._HEALTH_TIMEOUT),
                                self._HEALTH_TIMEOUT)
                            health.raise_for_status()
                            failures = 0
                        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
                            # Fresh SSE wins over a simultaneous slow/failed health endpoint.
                            if reader.done() or last_line > before_probe:
                                failures = 0
                                continue
                            failures += 1
                            if failures >= self._HEALTH_FAILURES:
                                raise ModelError('vLLM stream is silent and repeated read-only '
                                                 f'health checks failed: {type(exc).__name__}') from exc
                    await reader
                finally:
                    if not reader.done():
                        reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)

        def run() -> None:
            try:
                asyncio.run(receive())
            except asyncio.CancelledError:
                if not stopped.is_set():
                    events.put(('error', ModelError('vLLM stream reader was cancelled unexpectedly')))
            except Exception as exc:
                events.put(('error', exc if isinstance(exc, ModelError) else
                            ModelError(f'vLLM request failed: {type(exc).__name__}: {exc}')))
            finally:
                events.put(('done', None))

        # Check cancellation before issuing any request, including cancellation races
        # between creating this iterator and the first iteration.
        if on_tick:
            on_tick()
        worker = threading.Thread(target=run, name='aurex-vllm-stream', daemon=True)
        worker.start()
        next_tick = time.monotonic() + self._TICK_INTERVAL
        try:
            while True:
                now = time.monotonic()
                if now >= next_tick:
                    if on_tick:
                        on_tick()
                    next_tick = now + self._TICK_INTERVAL
                try:
                    kind, value = events.get(timeout=max(0.001, next_tick - time.monotonic()))
                except queue.Empty:
                    continue
                if kind == 'error':
                    raise value
                if kind == 'done':
                    return
                yield value
        finally:
            stopped.set()
            loop, task = state.get('loop'), state.get('task')
            if loop and task:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass  # Already completed and closed its event loop.
            # Cancelling async I/O closes the response/socket, rather than leaving a
            # blocked requests thread (and server generation) behind after Stop.
            worker.join(timeout=self._CLEANUP_TIMEOUT)

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             thinking: bool | None = None,
             max_tokens: int | None | object = _DEFAULT_MAX_TOKENS,
             generation_timeout_sec: float | None = None,
             on_delta: Callable[[str, str], None] | None = None,
             on_tick: Callable[[], None] | None = None) -> ModelReply:
        if (generation_timeout_sec is not None and
                (isinstance(generation_timeout_sec, bool) or
                 not isinstance(generation_timeout_sec, (int, float)) or
                 generation_timeout_sec <= 0)):
            raise ValueError('generation_timeout_sec must be a positive number or None')
        effective_thinking = self.config.enable_thinking if thinking is None else thinking
        payload = {'model': self.config.model, 'messages': messages,
                   'stream': True, 'stream_options': {'include_usage': True},
                   'chat_template_kwargs': self._template_kwargs(effective_thinking)}
        # Qwen's shipped generation_config is part of its reasoning recipe.
        # A low application temperature made long thinking responses repeatedly
        # revisit the same branch even though the stream was neither stalled nor
        # mechanically repetitive.  Preserve the configured deterministic
        # temperature for no-thinking controller/summary turns, while allowing
        # thinking turns to use the model-native sampler.  This is deliberately
        # not a token or wall-clock bound: difficult first turns may still reason
        # for as long as the task/context limits allow.
        if effective_thinking:
            thinking_frequency_penalty = getattr(
                self.config, 'thinking_frequency_penalty', 0.0)
            if thinking_frequency_penalty:
                payload['frequency_penalty'] = thinking_frequency_penalty
        else:
            payload['temperature'] = self.config.temperature
        requested_limit = (self.config.max_output_tokens
                           if max_tokens is _DEFAULT_MAX_TOKENS else max_tokens)
        if requested_limit is not None:
            payload['max_tokens'] = requested_limit
        if tools:
            payload.update(tools=tools, tool_choice='auto', parallel_tool_calls=False)
        else:
            # Summary/review calls must not reinterpret quoted historical tool
            # syntax as executable calls through an auto-tool parser.
            payload['tool_choice'] = 'none'
        progress = None
        if (getattr(self.config, 'stream_token_progress', False) is True and tools and
                payload['chat_template_kwargs']['enable_thinking'] is False):
            payload['return_token_ids'] = True
            progress = _TokenProgress()
        content, reasoning = [], []
        calls: dict[int, dict] = {}
        usage: dict = {}
        finish = ''
        stream_done = False
        on_tick = on_tick if on_tick is not None else getattr(self, 'on_tick', None)
        request_started = time.monotonic()
        generation_activity_started = None

        def partial_reply(finish_reason: str) -> ModelReply:
            # A partial call is retained only for private audit/recovery.  The
            # caller must never execute it because its JSON may end mid-token.
            return ModelReply(
                ''.join(content), ''.join(reasoning),
                [calls[i] for i in sorted(calls)], dict(usage), finish_reason)

        def generation_bound_progress(reason: str, now: float) -> dict:
            # Never expose argument text, call IDs, model text or token IDs in
            # liveness telemetry.  Parallel tool calls are disabled, but sum
            # argument characters defensively if a provider streams several.
            tool_name = ''
            argument_characters = 0
            for item in calls.values():
                function = item.get('function') or {}
                name = function.get('name')
                if not tool_name and isinstance(name, str) and name:
                    tool_name = name[:128]
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    argument_characters += len(arguments)
            activity_started = (generation_activity_started
                                if generation_activity_started is not None else now)
            return {
                'reason': reason,
                'tool_name': tool_name,
                'tool_argument_characters': argument_characters,
                'elapsed_seconds': round(max(0.0, now - activity_started), 3),
                'request_elapsed_seconds': round(max(0.0, now - request_started), 3),
            }

        def transport_tick() -> None:
            # User cancellation and the task's 1800s deadline remain
            # authoritative over this much smaller per-response navigation
            # bound.
            if on_tick:
                on_tick()
            now = time.monotonic()
            if (generation_timeout_sec is not None and
                    generation_activity_started is not None and
                    now - generation_activity_started >= generation_timeout_sec):
                report = generation_bound_progress('generation_wall_timeout', now)
                raise DegenerateGeneration(
                    'Current bounded model response reached its wall timeout; '
                    'its partial tool call was not executed',
                    partial_reply('generation_timeout'), report)

        lines = self._stream_lines(payload, on_tick=transport_tick)
        try:
            for line in lines:
                # Test the wall bound even for transports that do not implement
                # periodic on_tick callbacks, and before accepting another SSE
                # fragment after the deadline.
                transport_tick()
                if not line or not line.startswith(b'data:'):
                    continue
                raw = line[5:].strip()
                if raw == b'[DONE]':
                    stream_done = True
                    break
                packet = json.loads(raw)
                if packet.get('error'):
                    raise ModelError(str(packet['error']))
                if packet.get('usage'):
                    usage = packet['usage']
                for choice in packet.get('choices', []):
                    delta = choice.get('delta') or {}
                    has_activity = bool(
                        choice.get('token_ids') or choice.get('finish_reason') or
                        delta.get('content') or delta.get('reasoning_content') or
                        delta.get('reasoning') or delta.get('tool_calls'))
                    if has_activity and generation_activity_started is None:
                        generation_activity_started = time.monotonic()
                    # n defaults to one. Count only this request's generated
                    # choice-zero delta IDs, never prompt_token_ids or logprobs.
                    choice_index = choice.get('index', 0)
                    if progress is not None and type(choice_index) is int and choice_index == 0:
                        progress.observe(choice.get('token_ids'))
                        degenerate = progress.degenerate_reason()
                        # Repeated source code, tables and arrays are legal
                        # tool arguments. Token periodicity alone cannot tell
                        # a large HDL payload from broken natural language.
                        in_tool_payload = bool(calls or (choice.get('delta') or {}).get('tool_calls'))
                        if degenerate is not None and not in_tool_payload:
                            update = progress.take(final=True)
                            report = json.loads(update) if update is not None else {}
                            if update is not None and on_delta:
                                on_delta('progress', update)
                            partial = ModelReply(
                                ''.join(content), ''.join(reasoning),
                                [calls[i] for i in sorted(calls)], usage,
                                'repetition_guard')
                            raise DegenerateGeneration(
                                'Current model response hit the extreme mechanical repetition guard; '
                                'its partial tool call was not executed', partial,
                                {**report, 'reason': degenerate})
                        update = progress.take()
                        if update is not None and on_delta:
                            on_delta('progress', update)
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
                    for name, target in [('content', content), ('reasoning_content', reasoning), ('reasoning', reasoning)]:
                        if delta.get(name):
                            text = str(delta[name])
                            target.append(text)
                            if on_delta:
                                on_delta('text' if name == 'content' else 'reasoning', text)
                    for call in delta.get('tool_calls') or []:
                        item = calls.setdefault(int(call['index']), {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if call.get('id'):
                            item['id'] = call['id']
                        fn = call.get('function') or {}
                        for key in ('name', 'arguments'):
                            if fn.get(key):
                                item['function'][key] += fn[key]
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ModelError(f'vLLM request failed: {exc}') from exc
        finally:
            close = getattr(lines, 'close', None)
            if close:
                close()
        if progress is not None:
            update = progress.take(final=True)
            if update is not None and on_delta:
                on_delta('progress', update)
        if not stream_done:
            raise ModelError('vLLM stream ended without [DONE]; response is incomplete')
        if not finish:
            raise ModelError('vLLM stream ended without a finish reason; response is incomplete')
        if finish not in {'stop', 'tool_calls', 'length'}:
            raise ModelError(f'vLLM response did not complete normally ({finish}); no tool was executed')
        ordered = [calls[i] for i in sorted(calls)]
        generation_progress = (generation_bound_progress(
            'max_tokens', time.monotonic()) if finish == 'length' else None)
        reply = ModelReply(''.join(content), ''.join(reasoning), ordered, usage, finish,
                           generation_progress)
        if finish != 'length':
            if bool(ordered) != (finish == 'tool_calls'):
                raise InvalidToolCall('vLLM finish reason and tool calls disagree; no tool was executed', reply)
            seen = set()
            for call in ordered:
                fn = call['function']
                if (not isinstance(call['id'], str) or not call['id'].strip() or call['id'] in seen
                    or not isinstance(fn['name'], str) or not fn['name'].strip()):
                    raise InvalidToolCall('vLLM returned an incomplete or duplicate tool call identity; no tool was executed', reply)
                seen.add(call['id'])
                try:
                    arguments = json.loads(fn['arguments'])
                except (ValueError, TypeError) as exc:
                    raise InvalidToolCall('vLLM returned incomplete tool arguments; no tool was executed', reply) from exc
                if not isinstance(arguments, dict):
                    raise InvalidToolCall('vLLM tool arguments must be a JSON object; no tool was executed', reply)
        return reply
