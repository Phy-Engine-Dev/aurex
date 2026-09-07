# Aurex vision + reasoning service

This service is separate from the stopped development container `vllm`.
Both use physical GPUs 1 and 2; do not run them concurrently.

Start or replace **only the Aurex container**:

```sh
./scripts/aurex-vllm.sh
podman logs -f aurex-vllm
```

Stop without deleting either container:

```sh
podman stop aurex-vllm
```

The application connects to `http://127.0.0.1:8000/v1`, model `qwen38-27b`.
The API only binds the Linux loopback interface. A browser should use the
Aurex application, not connect directly to this model port.

## Request contract

- Total context: **90,112 tokens**, including prompt, image tokens and generated
  reasoning/answer. Reserve generation space before adding community posts,
  comments and experiment state; use `/tokenize` to measure text requests.
- At most **2 images per request**; video is disabled.
- Images are resized within `min_pixels=4096`, `max_pixels=589824` (768 × 768
  area). The processor preserves aspect ratio and rounds to its patch grid.
  This is a pixel-area cap, not a fixed 768-pixel limit on each edge.
- Vision encoder uses `TORCH_SDPA`, language decoding uses the existing SM70
  `FLASH_ATTN_V100` backend. Multimodal profiling is enabled.
- Thinking is **off by default**. The application enables it explicitly on the
  initial user-context turn with `"chat_template_kwargs": {"enable_thinking": true}`.
  Later agent/tool rounds use `false`. The `qwen3` parser places thinking into a separate
  response field. Do not concatenate that reasoning into saved assistant
  answer content or replay it as community source text.
- To explicitly disable thinking for a lightweight polling request, send
  `"chat_template_kwargs": {"enable_thinking": false}` in the API body.
- Tool calls use `qwen3_coder`; the application must validate tool arguments
  and execute tools itself.

The checkpoint's architecture supports a larger theoretical context, but this
deployment intentionally advertises its configured and memory-checked 90,112
limit. Do not substitute the checkpoint's theoretical maximum in application
budgeting.

The previous `gpu-memory-utilization=0.945` deployment, with full vision profiling (two maximum-size
images), allocation reported 3.47 GiB KV-cache memory per GPU and 107,240 KV
tokens. The configured 90,112-token context stayed below that measured capacity.
It is not a claim that a full 90K visual conversation has been benchmarked end
to end.

## Reproducible smoke test

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
podman exec -i aurex-vllm python3 - < scripts/aurex-vllm-smoke.py
# Include a 33K-token prompt and two maximum-size images:
podman exec -i -e AUREX_VLLM_STRESS=1 aurex-vllm python3 - < scripts/aurex-vllm-smoke.py
# Sample GPU use during thinking and no-thinking decoding:
podman exec -i aurex-vllm python3 - < scripts/aurex-vllm-benchmark.py
```

The smoke test draws a labelled 5 V / 10 ohm circuit and checks that the model
reads the values, obtains 0.5 A, and returns nonempty reasoning in a separate
field without leaking `<think>` delimiters into ordinary answer content.
It also checks automatic tool-call parsing and explicit per-request no-thinking.
It does not create an Aurex conversation or print the reasoning text.

## Observed results (2026-09-05)

These are historical short-probe results, not the current long-session stability
claim. On 2026-09-05 at 16:28:07 UTC, a long community replay exhausted CUDA
memory: custom all-reduce requested another 20 MiB with only 16.62 MiB free.
The worker then died and the API server exited. A container exit code of zero
and `OOMKilled=false` did not rule out this CUDA OOM.

The recovery configuration uses `gpu-memory-utilization=0.93` and
`--disable-custom-all-reduce`, leaving more runtime headroom while retaining the
same model, precision, 90,112-token context, vision limits, and batch size.
Startup profiling and the multimodal smoke test must pass again; the old
15.5 GiB / 45-token-per-second figures below do not describe this new setup.

### Recovery checks (2026-09-06, Asia/Shanghai)

- Startup with the recovery flags measured 3.21 GiB KV cache and 99,048 KV
  tokens, 1.10x the configured 90,112-token request length.
- Both physical V100s used 15,592 MiB immediately after startup. This is an
  observation, not a fixed allocation target or a long-run peak claim.
- The two-image thinking test completed with `stop` in 8.80 s, correctly
  answering 0.5 A; reasoning remained a separate response field.
- The no-thinking tool-call test passed with valid structured arguments.
- A 33,669-token prompt with two maximum-size images completed in 23.91 s.
- A separate text-only 88,977-token prompt completed with 2 output tokens,
  normal `stop` and SSE DONE in 86.446 s. Across 163 memory samples at a nominal
  0.5-second interval, both GPUs peaked at 15,712 MiB. This bounds only sampled
  observations for this short-output probe, not hours of decoding or reasoning
  quality at the context limit.
- The failed container was retained as `aurex-vllm-oom-20260905`; the development
  `vllm` container was neither removed nor started.

### Historical measurements before the CUDA OOM

- `/health` and `/v1/models` succeed; model advertises 90,112 total tokens.
- Two 768 × 768 images: correctly reads 5 V / 10 ohm and answers 0.5 A;
  300 generated tokens in 8.80 s including prefill, with 982 reasoning characters
  in a separate field and a normal `stop` finish.
- A 33,669-token prompt with two maximum-size images completes in 23.72 s.
- Memory after startup is 15,872 MiB (15.500 GiB) per GPU; the measured runtime
  peak and allocator-retained memory are 16,008 MiB (15.633 GiB) per GPU.
  This leaves 376 MiB physical headroom on each 16 GiB V100 for the tested image
  and batching limits; do not raise those limits without profiling again.
- Thinking decode: 45.18 tokens/s, both GPUs at 100% in all 21 sampled decode
  observations, average power 188.6 / 190.6 W.
- No-thinking decode: 45.68 tokens/s, both GPUs at 100% in all 12 sampled decode
  observations, average power 195.1 / 195.2 W.
- Long-prefill power peaks: 304.73 / 306.92 W from NVML sampling against a
  configured 300 W power limit. Brief telemetry peaks can exceed the nominal
  limit. Decode power need not reach 300 W to maintain full reported GPU use.

The thinking throughput probe intentionally has a 512-token output cap and can
finish with `length`; it measures decode performance, not answer correctness.
The separate circuit smoke test must finish with `stop`. These short measurements
do not promise constant 100% utilization for every future workload.

One earlier startup was killed by host `systemd-oomd` while an unrelated UWVM
compiler used about 30 GiB RAM and `/tmp` tmpfs occupied about 39 GiB. This was
host RAM pressure, not CUDA OOM. The final startup and tests succeeded after that
compiler exited. Avoid overlapping similarly large builds with model startup.

The old OpenCode binary, conversation database, and `vllm` container are not
modified by these scripts.
