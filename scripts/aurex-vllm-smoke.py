#!/usr/bin/env python3
"""Exercise the live multimodal API; run in aurex-vllm (Pillow is installed).

    podman exec -i aurex-vllm python3 - < scripts/aurex-vllm-smoke.py

This creates no conversations and does not print the model's reasoning text.
"""
import base64
import io
import json
import os
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

BASE_URL = os.environ.get("AUREX_VLLM_URL", "http://127.0.0.1:8000")


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


image = Image.new("RGB", (768, 768), "white")
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 30)
except OSError:
    font = ImageFont.load_default(size=30)
draw.text((190, 20), "IDEAL DC CIRCUIT", fill="black", font=font)
draw.line([(160, 200), (160, 130), (600, 130), (600, 225)], fill="black", width=5)
draw.line([(600, 305), (600, 390), (160, 390), (160, 280)], fill="black", width=5)
draw.line([(120, 220), (200, 220)], fill="black", width=6)
draw.line([(138, 250), (182, 250)], fill="black", width=6)
draw.line([(160, 200), (160, 220)], fill="black", width=5)
draw.line([(160, 250), (160, 280)], fill="black", width=5)
draw.rectangle((575, 225, 625, 305), outline="black", width=5)
draw.text((230, 215), "V1 = 5 V", fill="black", font=font)
draw.text((395, 250), "R1 = 10 ohm", fill="black", font=font)
png = io.BytesIO()
image.save(png, format="PNG")
url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()

models = request("/v1/models")
model = models["data"][0]
assert model["id"] == "qwen38-27b", model
assert model["max_model_len"] == 90112, model

started = time.monotonic()
result = request(
    "/v1/chat/completions",
    {
        "model": model["id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "image_url", "image_url": {"url": url}},
                    {
                        "type": "text",
                        "text": "These are two copies of the same circuit. Read R1 resistance and V1 voltage from the image, then calculate the ideal loop current. Give a concise answer with numbers and units.",
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)
message = result["choices"][0]["message"]
content = message.get("content") or ""
reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
assert reasoning, "Expected explicit thinking in a separate reasoning field"
assert "<think>" not in content and "</think>" not in content, message
assert "10" in content and "5" in content, content
assert "0.5" in content or "500" in content, content
assert result["choices"][0]["finish_reason"] == "stop", result["choices"][0]["finish_reason"]
print(json.dumps({
    "model": model["id"],
    "max_model_len": model["max_model_len"],
    "elapsed_seconds": round(time.monotonic() - started, 2),
    "vision_answer": content,
    "reasoning_chars": len(reasoning),
    "reasoning_separated": True,
    "images": 2,
    "pixels_per_image": 589824,
    "usage": result.get("usage"),
}, ensure_ascii=False, indent=2))

tool_result = request("/v1/chat/completions", {
    "model": model["id"],
    "messages": [{"role": "user", "content": "Use get_component_info to inspect component R1. Do not guess its value."}],
    "tools": [{"type": "function", "function": {
        "name": "get_component_info",
        "description": "Read the exact properties of a circuit component.",
        "parameters": {
            "type": "object",
            "properties": {"component_id": {"type": "string"}},
            "required": ["component_id"],
            "additionalProperties": False,
        },
    }}],
    "tool_choice": "auto",
    "chat_template_kwargs": {"enable_thinking": False},
    "max_tokens": 128,
    "temperature": 0,
})
tool_message = tool_result["choices"][0]["message"]
tool_calls = tool_message.get("tool_calls") or []
assert tool_calls, tool_message
assert tool_calls[0]["function"]["name"] == "get_component_info", tool_calls
assert json.loads(tool_calls[0]["function"]["arguments"])["component_id"] == "R1", tool_calls
assert not (tool_message.get("reasoning") or tool_message.get("reasoning_content")), tool_message
print(json.dumps({"tool_call_parsed": True, "per_request_no_think": True}))

if os.environ.get("AUREX_VLLM_STRESS") == "1":
    context = "\n".join(
        f"Record {i}: resistor R{i} connects node N{i} to node N{i+1}; resistance {i+10} ohm."
        for i in range(1000)
    )
    stress_content = [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": context + "\nAcknowledge receipt of this background dataset with the word READY only. Do not analyze it."},
    ]
    started = time.monotonic()
    stress = request("/v1/chat/completions", {
        "model": model["id"],
        "messages": [{"role": "user", "content": stress_content}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 16,
        "temperature": 0,
    })
    assert "READY" in (stress["choices"][0]["message"].get("content") or ""), stress
    assert stress["usage"]["prompt_tokens"] >= 16000, stress["usage"]
    print(json.dumps({
        "long_prefill_with_two_images": True,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "usage": stress["usage"],
    }))
