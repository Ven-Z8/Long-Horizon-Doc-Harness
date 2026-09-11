#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_DIR="${RUNPOD_WORKSPACE_DIR:-/workspace}"
REPO_DIR="${RUNPOD_REPO_DIR:-${WORKSPACE_DIR}/Long-Horizon-Doc-Harness}"
MODELS_DIR="${RUNPOD_MODELS_DIR:-${REPO_DIR}/models}"
DOCUMENTS_DIR="${RUNPOD_DOCUMENTS_DIR:-${REPO_DIR}/data/documents}"
BENCHMARK_DIR="${RUNPOD_BENCHMARK_DIR:-${REPO_DIR}/benchmark/mmlongbench-doc-v2}"
MIN_GPU_MIB="${RUNPOD_MIN_GPU_MIB:-22000}"
CHECK_TESTS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--repo-dir PATH] [--check-tests]

Validate the RunPod CUDA environment, pinned assets, and benchmark checkout.
The OpenRouter key is checked only for presence; its value is never printed.
EOF
}

log() { printf '[runpod-verify] %s\n' "$*"; }
fail() { printf '[runpod-verify] ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --repo-dir) REPO_DIR="${2:?missing path after --repo-dir}"; shift 2 ;;
    --check-tests) CHECK_TESTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
MODELS_DIR="${RUNPOD_MODELS_DIR:-${REPO_DIR}/models}"
DOCUMENTS_DIR="${RUNPOD_DOCUMENTS_DIR:-${REPO_DIR}/data/documents}"
BENCHMARK_DIR="${RUNPOD_BENCHMARK_DIR:-${REPO_DIR}/benchmark/mmlongbench-doc-v2}"

[[ -d "$REPO_DIR/.git" ]] || fail "not a Git checkout: $REPO_DIR"
command -v uv >/dev/null 2>&1 || fail "uv is not installed"

log "checkout: $(git -C "$REPO_DIR" branch --show-current) @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
log "disk: $(df -h "$WORKSPACE_DIR" | awk 'NR==2 {print $4 " free of " $2}')"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi could not query the GPU"
else
  fail "nvidia-smi is not installed; use a CUDA RunPod template"
fi

CUDA_REPORT="$(cd "$REPO_DIR" && uv run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu_{index}={props.name};memory_mib={props.total_memory // (1024 * 1024)}")
PY
)" || fail "PyTorch CUDA probe failed"
printf '%s\n' "$CUDA_REPORT"
grep -q 'cuda_available=True' <<<"$CUDA_REPORT" || fail "PyTorch cannot see CUDA"
GPU_MIB="$(sed -n 's/.*memory_mib=\([0-9][0-9]*\).*/\1/p' <<<"$CUDA_REPORT" | head -n 1)"
[[ -n "$GPU_MIB" && "$GPU_MIB" -ge "$MIN_GPU_MIB" ]] || fail "GPU memory is ${GPU_MIB:-unknown} MiB; need at least ${MIN_GPU_MIB} MiB"

for model in Qwen3.5-4B Qwen3-VL-Embedding-2B Qwen3-VL-Reranker-2B Qianfan-OCR; do
  [[ -f "$MODELS_DIR/$model/config.json" ]] || fail "missing model config: $MODELS_DIR/$model"
  compgen -G "$MODELS_DIR/$model/*.safetensors" >/dev/null || fail "missing model weights: $MODELS_DIR/$model"
  log "model present: $model"
done

PDF_COUNT="$(find "$DOCUMENTS_DIR" -maxdepth 1 -type f -name '*.pdf' 2>/dev/null | wc -l)"
[[ "$PDF_COUNT" -ge 135 ]] || fail "found ${PDF_COUNT} PDFs; expected at least 135 in ${DOCUMENTS_DIR}"
log "documents: ${PDF_COUNT} PDFs"

SAMPLE_COUNT="$(cd "$REPO_DIR" && uv run python - <<'PY'
import json
from pathlib import Path
path = Path("benchmark/mmlongbench-doc-v2/data/samples.json")
print(len(json.loads(path.read_text(encoding="utf-8"))))
PY
)" || fail "could not read V2 samples"
[[ "$SAMPLE_COUNT" -eq 1071 ]] || fail "found ${SAMPLE_COUNT} V2 samples; expected 1071"
[[ -f "$BENCHMARK_DIR/eval/evaluate.py" ]] || fail "V2 evaluator is missing"
log "benchmark: ${SAMPLE_COUNT} samples and evaluator present"

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  log "OpenRouter key: present in environment"
elif [[ -f "$REPO_DIR/.env" ]] && grep -Eq '^OPENROUTER_API_KEY=[^[:space:]]+' "$REPO_DIR/.env"; then
  log "OpenRouter key: present in ${REPO_DIR}/.env"
else
  fail "OPENROUTER_API_KEY is not present in the environment or .env"
fi

if (( CHECK_TESTS )); then
  (cd "$REPO_DIR" && uv run pytest -q)
fi

log "environment verification passed"
