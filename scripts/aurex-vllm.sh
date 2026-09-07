#!/usr/bin/env bash
set -euo pipefail

# Read-only status and scoped stop must never recreate the container.
case "${1:-start}" in
  status)
    podman ps -a --filter 'name=^aurex-vllm$' --format '{{.Names}} {{.Status}}'
    if [[ "$(podman inspect aurex-vllm --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]; then
      curl --noproxy '*' --fail --silent --max-time 5 http://127.0.0.1:8000/health
      echo
    fi
    exit 0
    ;;
  stop)
    podman stop --time 60 aurex-vllm
    echo 'Stopped aurex-vllm; container, image and model files retained.'
    exit 0
    ;;
  start)
    if [[ "$(podman inspect aurex-vllm --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]; then
      echo 'aurex-vllm is already running. Stop it explicitly before applying changed parameters.'
      exit 0
    fi
    ;;
  *) echo "Usage: $0 [start|status|stop]" >&2; exit 2 ;;
esac

# Separate multimodal service. Never changes the development `vllm` container.
# Stop that container before starting this one: both use physical GPUs 1 and 2.
if [[ "$(podman inspect vllm --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]; then
  echo 'The development vllm container is still running; stop it first.' >&2
  exit 1
fi

exec podman run -d --replace --name aurex-vllm \
  --entrypoint vllm --ipc=host \
  --device nvidia.com/gpu=1 --device nvidia.com/gpu=2 \
  -p 127.0.0.1:8000:8000 \
  -v /home/macromodel/.cache/huggingface:/root/.cache/huggingface \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -e VLLM_SM70_QUANT_BACKEND=turbomind \
  -e VLLM_SM70_FLASH_ATTN_V100=1 \
  -e VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1 \
  -e VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE=1 \
  -e VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=0 \
  -e VLLM_SM70_QWEN_GDN_FULL_FORWARD=0 \
  -e VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD=0 \
  -e VLLM_SM70_QWEN_GDN_SPEC_CORE_OP=0 \
  -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= \
  -e http_proxy= -e https_proxy= -e all_proxy= \
  -e NO_PROXY=127.0.0.1,localhost -e no_proxy=127.0.0.1,localhost \
  localhost/cat1-vllm:v1.5-sm70-patched \
  serve shawnw3i/Qwen3.8-27B-AWQ-MTP \
  --tokenizer Qwen/Qwen3.8-27B \
  --served-model-name qwen38-27b \
  --tensor-parallel-size 2 --quantization awq --dtype float16 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 90112 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":4096,"max_pixels":589824}' \
  --mm-processor-cache-gb 0 --mm-encoder-attn-backend TORCH_SDPA \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --attention-backend FLASH_ATTN_V100 \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"full_decode_only","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 --port 8000
