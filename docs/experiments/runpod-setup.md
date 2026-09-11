# RunPod environment setup

This is the handoff procedure for the long-document harness. It is designed for
an RTX 4090 pod with a persistent volume mounted at `/workspace`. The pod is
disposable; the checkout, models, PDFs, benchmark files, caches, and artifacts
are stored on the volume.

## 1. Create or reconnect to the pod

Choose a CUDA-enabled RunPod template with at least 24 GB VRAM and attach the
persistent volume at `/workspace`. A 4090 is the current target. Open an SSH
terminal or the web terminal and check the mount:

```bash
df -h /workspace
nvidia-smi
```

Do not store the repository under `/tmp` or a non-persistent container path.

## 2. Put the committed branch on the volume

The repository branch used for the current implementation is
`feature/mmlongbench-v2-baseline`. Replace the URL with your GitHub repository:

```bash
cd /workspace
git clone --branch feature/mmlongbench-v2-baseline \
  https://github.com/<account>/Long-Horizon-Doc-Harness.git \
  Long-Horizon-Doc-Harness
cd /workspace/Long-Horizon-Doc-Harness
```

If the directory already exists, do not clone over it. Check the saved commit
and branch instead:

```bash
git status --short --branch
git log -1 --oneline
```

The current implementation commit is `2f0d4a5`. The large assets are ignored by
Git and are downloaded by the next step.

## 3. Bootstrap dependencies and assets

Run the idempotent bootstrap. It installs the project extras, downloads the four
pinned checkpoints from `models.lock`, fetches the 135 PDFs, and checks out the
pinned V2 evaluator and its 1,071 samples:

```bash
cd /workspace/Long-Horizon-Doc-Harness
bash scripts/runpod/bootstrap.sh \
  --skip-repo \
  --branch feature/mmlongbench-v2-baseline
```

The script reuses complete existing assets and never deletes `models/`,
`data/documents/`, `benchmark/`, `cache/`, or `artifacts/`. The Hugging Face
cache is placed at `/workspace/.cache/huggingface`, and the uv cache is placed at
`/workspace/.cache/uv` so both survive pod replacement.

If the pod image does not have `uv`, the script installs it under
`/workspace/.uv/bin`. If a model requires Hugging Face authentication, set
`HF_TOKEN` in the terminal before rerunning; the token is never written to the
repository.

## 4. Add the judge credential safely

The local model stages do not need OpenRouter. The key is only for the external
benchmark judge. Create the ignored `.env` with mode `600`:

```bash
cd /workspace/Long-Horizon-Doc-Harness
umask 077
cat > .env <<'__ENV__'
OPENROUTER_API_KEY=<paste-your-key-here>
__ENV__
```

Never commit `.env`, print it, or put the key in a command history. The
verification script reports only whether the key is present.

## 5. Verify before spending GPU time

```bash
cd /workspace/Long-Horizon-Doc-Harness
bash scripts/runpod/verify.sh --check-tests
```

The check confirms the checkout, GPU visibility, PyTorch CUDA support, at least
22 GB of GPU memory, all four local checkpoints, 135 PDFs, 1,071 samples, the
evaluator, and the presence of the OpenRouter key. A failure is a setup issue;
do not start a long run until it is resolved.

## 6. Run a one-question GPU probe

Use one rendered page from a small PDF to validate the answer model and image
processor. The command writes only ignored artifacts:

```bash
mkdir -p artifacts/runpod-probe
uv run python -m doc_harness.cli gpu-smoke \
  --config configs/baseline.toml \
  --question 'What is shown on this page?' \
  --image cache/runpod-probe-page.png \
  --output artifacts/runpod-probe/records.jsonl \
  --prediction artifacts/runpod-probe/predictions.json
```

If no PNG exists yet, render a page first:

```bash
uv run python - <<'PY'
from pathlib import Path
from doc_harness.rendering import render_pdf

pages = render_pdf(
    Path('data/documents/welcome-to-nus.pdf'),
    Path('cache/runpod-probe'),
    dpi=96,
)
print(pages[0].image_path)
PY
```

Pass the printed image path to `gpu-smoke`.

## 7. Resume the benchmark work

Prepare the frozen 100-question selection if it is not already on the volume:

```bash
uv run python -m doc_harness.cli prepare-phase2 \
  --samples benchmark/mmlongbench-doc-v2/data/samples.json \
  --baseline-run artifacts/runs/B5-dev-100 \
  --output-dir artifacts/splits/phase2
```

Then follow `docs/experiments/phase2-runbook.md`. Start with the 20-question
smoke pair, inspect checkpoints, and only then run the matched 100-question
control and prompt variants. Each phase loads one heavyweight model, writes
durable artifacts, and exits before the next phase starts.

To resume after a pod restart, reconnect the same volume and repeat the exact
run command with `--resume`. Do not change the config, prompt set, model
revision, samples, or code identity inside an existing run directory.

## 8. Shut down without losing progress

Before stopping the pod, wait for the command to exit and confirm the run files
are present:

```bash
git status --short --branch
find artifacts/runs -maxdepth 2 -type f | sort | tail -30
du -sh models data/documents benchmark cache artifacts
```

RunPod can then be stopped. The persistent volume retains the checkout, models,
PDFs, caches, and committed run checkpoints; the GPU container itself does not
need to remain running.
