#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible, resumable setup for a RunPod persistent volume. The script
# never removes models, documents, benchmark files, cache entries, or artifacts.

SCRIPT_NAME="$(basename "$0")"
WORKSPACE_DIR="${RUNPOD_WORKSPACE_DIR:-/workspace}"
REPO_DIR="${RUNPOD_REPO_DIR:-${WORKSPACE_DIR}/Long-Horizon-Doc-Harness}"
REPO_URL="${RUNPOD_REPO_URL:-}"
BRANCH="${RUNPOD_BRANCH:-feature/mmlongbench-v2-baseline}"
MODELS_DIR="${RUNPOD_MODELS_DIR:-${REPO_DIR}/models}"
DOCUMENTS_DIR="${RUNPOD_DOCUMENTS_DIR:-${REPO_DIR}/data/documents}"
BENCHMARK_DIR="${RUNPOD_BENCHMARK_DIR:-${REPO_DIR}/benchmark/mmlongbench-doc-v2}"
HF_HOME_DIR="${HF_HOME:-${WORKSPACE_DIR}/.cache/huggingface}"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-${WORKSPACE_DIR}/.cache/uv}"

SKIP_REPO=0
SKIP_DEPENDENCIES=0
SKIP_MODELS=0
SKIP_DATA=0
SKIP_TESTS=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [options]

Prepare a RunPod persistent-volume checkout for the long-document harness.
Existing assets are kept and reused; nothing in models/, data/, benchmark/,
cache/, or artifacts/ is deleted.

Options:
  --repo-url URL          GitHub URL used only when --repo-dir is absent
  --branch NAME           branch to clone (default: ${BRANCH})
  --repo-dir PATH         repository path (default: ${REPO_DIR})
  --skip-repo             do not clone or inspect the repository checkout
  --skip-dependencies     do not install the uv environment
  --skip-models           do not download Hugging Face checkpoints
  --skip-data             do not download PDFs or the V2 evaluator
  --skip-tests            do not run the offline test suite
  --dry-run               print actions without changing the filesystem
  -h, --help              show this help

Environment overrides:
  RUNPOD_WORKSPACE_DIR, RUNPOD_REPO_DIR, RUNPOD_REPO_URL, RUNPOD_BRANCH,
  RUNPOD_MODELS_DIR, RUNPOD_DOCUMENTS_DIR, RUNPOD_BENCHMARK_DIR, HF_TOKEN.
EOF
}

log() { printf '[runpod] %s\n' "$*"; }
die() { printf '[runpod] ERROR: %s\n' "$*" >&2; exit 1; }

run() {
  printf '[runpod] +'
  printf ' %q' "$@"
  printf '\n'
  if (( ! DRY_RUN )); then
    "$@"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

parse_args() {
  while (($#)); do
    case "$1" in
      --repo-url) REPO_URL="${2:?missing URL after --repo-url}"; shift 2 ;;
      --branch) BRANCH="${2:?missing branch after --branch}"; shift 2 ;;
      --repo-dir) REPO_DIR="${2:?missing path after --repo-dir}"; shift 2 ;;
      --skip-repo) SKIP_REPO=1; shift ;;
      --skip-dependencies) SKIP_DEPENDENCIES=1; shift ;;
      --skip-models) SKIP_MODELS=1; shift ;;
      --skip-data) SKIP_DATA=1; shift ;;
      --skip-tests) SKIP_TESTS=1; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown option: $1" ;;
    esac
  done
  MODELS_DIR="${RUNPOD_MODELS_DIR:-${REPO_DIR}/models}"
  DOCUMENTS_DIR="${RUNPOD_DOCUMENTS_DIR:-${REPO_DIR}/data/documents}"
  BENCHMARK_DIR="${RUNPOD_BENCHMARK_DIR:-${REPO_DIR}/benchmark/mmlongbench-doc-v2}"
}

clone_at_commit() {
  local url="$1" commit="$2" destination
  destination="$(mktemp -d "${WORKSPACE_DIR}/.runpod-source.XXXXXX")"
  run git init -q "$destination"
  run git -C "$destination" remote add origin "$url"
  run git -C "$destination" fetch --depth 1 origin "$commit"
  run git -C "$destination" checkout --detach FETCH_HEAD
  if command -v git-lfs >/dev/null 2>&1; then
    run git -C "$destination" lfs pull
  fi
  CLONED_DIR="$destination"
}

cleanup_clone() {
  if [[ -n "${CLONED_DIR:-}" && -d "${CLONED_DIR}" ]]; then
    if (( DRY_RUN )); then
      log "would remove temporary checkout ${CLONED_DIR}"
    else
      rm -rf -- "${CLONED_DIR}"
    fi
  fi
  CLONED_DIR=""
}

setup_repo() {
  if (( SKIP_REPO )); then
    log "repository step skipped"
    return
  fi
  require_command git
  if [[ -d "${REPO_DIR}/.git" ]]; then
    log "using existing checkout at ${REPO_DIR}"
    local current_branch
    current_branch="$(git -C "$REPO_DIR" branch --show-current 2>/dev/null || true)"
    if [[ -n "$current_branch" && "$current_branch" != "$BRANCH" ]]; then
      log "warning: checkout is on ${current_branch}, requested ${BRANCH}; no pull or checkout was performed"
    fi
    return
  fi
  [[ -n "$REPO_URL" ]] || die "${REPO_DIR} is absent; provide --repo-url URL or clone the repository first"
  run mkdir -p "$(dirname "$REPO_DIR")"
  run git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
}

setup_dependencies() {
  if (( SKIP_DEPENDENCIES )); then
    log "dependency step skipped"
    return
  fi
  require_command git
  if ! command -v uv >/dev/null 2>&1; then
    require_command curl
    local uv_install_dir="${UV_INSTALL_DIR:-${WORKSPACE_DIR}/.uv/bin}"
    run mkdir -p "$uv_install_dir"
    if (( ! DRY_RUN )); then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_install_dir" sh
      export PATH="${uv_install_dir}:$PATH"
    fi
  fi
  command -v uv >/dev/null 2>&1 || (( DRY_RUN )) || die "uv is not available after installation"
  export UV_PROJECT_ENVIRONMENT="${REPO_DIR}/.venv"
  export UV_CACHE_DIR="${UV_CACHE_DIR_VALUE}"
  run mkdir -p "$UV_CACHE_DIR"
  run bash -c "cd \"$REPO_DIR\" && uv sync --extra dev --extra pdf --extra gpu --extra eval"
  run bash -c "cd \"$REPO_DIR\" && uv pip install -e ."
}

lock_value() {
  local key="$1"
  sed -n -E "s/^${key}[[:space:]]*=[[:space:]]*//p" "${REPO_DIR}/models.lock" | head -n 1
}

download_model() {
  local key="$1" target="$2" spec model_id revision
  spec="$(lock_value "$key")"
  [[ "$spec" == *@* ]] || die "invalid ${key} entry in models.lock"
  model_id="${spec%@*}"
  revision="${spec##*@}"
  if (( ! DRY_RUN )); then
    mkdir -p "$target"
  fi
  if [[ -f "${target}/config.json" ]] && compgen -G "${target}/*.safetensors" >/dev/null; then
    log "reusing existing checkpoint ${model_id} at ${target}"
    return
  fi
  local hf_args=(download "$model_id" --revision "$revision" --local-dir "$target")
  if command -v hf >/dev/null 2>&1; then
    run hf "${hf_args[@]}"
  else
    command -v uvx >/dev/null 2>&1 || die "hf is missing and uvx is unavailable; install huggingface_hub"
    run uvx --from huggingface_hub hf "${hf_args[@]}"
  fi
  if (( ! DRY_RUN )); then
    printf '%s\n' "$revision" > "${target}/.runpod-revision"
  fi
}

setup_models() {
  if (( SKIP_MODELS )); then
    log "model download step skipped"
    return
  fi
  [[ -f "${REPO_DIR}/models.lock" ]] || die "models.lock is missing from ${REPO_DIR}"
  export HF_HOME="${HF_HOME_DIR}"
  run mkdir -p "$HF_HOME" "$MODELS_DIR"
  download_model reasoner "${MODELS_DIR}/Qwen3.5-4B"
  download_model retriever "${MODELS_DIR}/Qwen3-VL-Embedding-2B"
  download_model reranker "${MODELS_DIR}/Qwen3-VL-Reranker-2B"
  download_model ocr "${MODELS_DIR}/Qianfan-OCR"
}

setup_documents() {
  if [[ -f "${DOCUMENTS_DIR}/.source-commit" && "$(cat "${DOCUMENTS_DIR}/.source-commit")" == "$DOC_COMMIT" \
    && "$(find "$DOCUMENTS_DIR" -maxdepth 1 -type f -name '*.pdf' | wc -l)" -ge 135 ]]; then
    log "reusing the 135-document MMLongBench-Doc checkout"
    return
  fi
  run mkdir -p "$DOCUMENTS_DIR"
  if (( DRY_RUN )); then
    log "would fetch ${DOC_URL}@${DOC_COMMIT} and copy PDFs to ${DOCUMENTS_DIR}"
    return
  fi
  clone_at_commit "$DOC_URL" "$DOC_COMMIT"
  local count
  count="$(find "$CLONED_DIR" -type f -iname '*.pdf' | wc -l)"
  (( count >= 135 )) || die "document checkout contains ${count} PDFs; expected at least 135"
  while IFS= read -r -d '' pdf; do
    cp -n "$pdf" "${DOCUMENTS_DIR}/$(basename "$pdf")"
  done < <(find "$CLONED_DIR" -type f -iname '*.pdf' -print0)
  printf '%s\n' "$DOC_COMMIT" > "${DOCUMENTS_DIR}/.source-commit"
  cleanup_clone
}

setup_benchmark() {
  if [[ -f "${BENCHMARK_DIR}/.source-commit" && "$(cat "${BENCHMARK_DIR}/.source-commit")" == "$EVAL_COMMIT" \
    && -f "${BENCHMARK_DIR}/data/samples.json" && -f "${BENCHMARK_DIR}/eval/evaluate.py" ]]; then
    log "reusing the pinned MMLongBench-Doc V2 evaluator"
    return
  fi
  run mkdir -p "${BENCHMARK_DIR}/data" "${BENCHMARK_DIR}/eval"
  if (( DRY_RUN )); then
    log "would fetch ${EVAL_URL}@${EVAL_COMMIT} into ${BENCHMARK_DIR}"
    return
  fi
  clone_at_commit "$EVAL_URL" "$EVAL_COMMIT"
  [[ -f "${CLONED_DIR}/data/samples.json" ]] || die "V2 checkout has no data/samples.json"
  [[ -f "${CLONED_DIR}/eval/evaluate.py" ]] || die "V2 checkout has no eval/evaluate.py"
  cp -n "${CLONED_DIR}/data/"* "${BENCHMARK_DIR}/data/"
  cp -n "${CLONED_DIR}/eval/"* "${BENCHMARK_DIR}/eval/"
  printf '%s\n' "$EVAL_COMMIT" > "${BENCHMARK_DIR}/.source-commit"
  cleanup_clone
}

setup_data() {
  if (( SKIP_DATA )); then
    log "benchmark/data step skipped"
    return
  fi
  export GIT_TERMINAL_PROMPT=0
  DOC_URL="https://github.com/mayubo2333/MMLongBench-Doc.git"
  DOC_COMMIT="d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384"
  EVAL_URL="https://github.com/VectifyAI/MMLongBench-Doc-V2.git"
  EVAL_COMMIT="3aba3c6831a432ee882f763435f9ebe92ab75ed9"
  require_command git
  setup_documents
  setup_benchmark
}

run_tests() {
  if (( SKIP_TESTS )); then
    log "test step skipped"
    return
  fi
  run bash -c "cd \"$REPO_DIR\" && uv run pytest -q"
}

main() {
  parse_args "$@"
  log "persistent workspace: ${WORKSPACE_DIR}"
  log "repository: ${REPO_DIR}"
  if (( DRY_RUN )); then log "dry-run mode: no changes will be made"; fi
  setup_repo
  setup_dependencies
  setup_models
  setup_data
  run_tests
  log "bootstrap complete; run scripts/runpod/verify.sh from the repository"
}

main "$@"
