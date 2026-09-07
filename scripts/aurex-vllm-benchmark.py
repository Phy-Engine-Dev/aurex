#!/usr/bin/env python3
"""Sample both GPUs during thinking and no-thinking streaming generation.

    podman exec -i aurex-vllm python3 - < scripts/aurex-vllm-benchmark.py

Reports measurements, not a promise of sustained 100% utilization or 300 W.
No model text or reasoning is printed. Requests create no app conversations.
"""
import json
import statistics
import subprocess
import threading
import time
import urllib.request


def measure(thinking):
    samples = []
    done = threading.Event()

    def sample():
        while not done.is_set():
            output = subprocess.run([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, check=True).stdout
            at = time.monotonic()
            for line in output.splitlines():
                index, memory, util, power = [float(value.strip()) for value in line.split(",")]
                samples.append((at, int(index), memory, util, power))
            done.wait(0.5)

    payload = {
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": (
            "A 5 V source feeds a 10 ohm resistor in series with a parallel combination "
            "of 20 ohm and 30 ohm. Calculate total current, branch currents, and power "
            "in each resistor. Explain the calculation carefully with units."
            if thinking else "Output a JSON array containing the integers from 1 through 100 inclusive. Output only the array."
        )}],
        "chat_template_kwargs": {"enable_thinking": thinking},
        "temperature": 0,
        "max_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.monotonic()
    first = None
    last = None
    usage = None
    reasoning_chars = 0
    content_chars = 0
    finish_reason = None
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    reason = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""
                    if reason or content:
                        last = time.monotonic()
                        if first is None:
                            first = last
                    reasoning_chars += len(reason)
                    content_chars += len(content)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    finally:
        done.set()
        sampler.join(timeout=5)
    assert first is not None and last is not None, "No generated tokens"
    assert bool(reasoning_chars) == thinking, "Thinking flag did not match reasoning channel"
    gpus = {}
    for index in sorted({row[1] for row in samples}):
        rows = [row for row in samples if row[1] == index and first <= row[0] <= last]
        if not rows:
            continue
        gpus[index] = {
            "samples": len(rows),
            "memory_peak_mib": max(row[2] for row in samples if row[1] == index),
            "decode_util_avg_percent": round(statistics.mean(row[3] for row in rows), 1),
            "decode_util_min_percent": min(row[3] for row in rows),
            "decode_util_max_percent": max(row[3] for row in rows),
            "decode_power_avg_w": round(statistics.mean(row[4] for row in rows), 1),
            "decode_power_max_w": max(row[4] for row in rows),
        }
    print(json.dumps({
        "thinking": thinking,
        "first_token_seconds": round(first - start, 3),
        "decode_seconds": round(last - first, 3),
        "decode_tokens_per_second": round((usage["completion_tokens"] - 1) / max(last - first, 0.001), 2) if usage else None,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "finish_reason": finish_reason,
        "usage": usage,
        "gpus": gpus,
    }, ensure_ascii=False), flush=True)


measure(True)
measure(False)
