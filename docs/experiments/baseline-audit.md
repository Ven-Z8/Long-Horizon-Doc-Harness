# Historical V2 baseline audit

This note records the audit of the interrupted 103-question baseline. The source
artifacts remain unchanged under `artifacts/`; the ignored snapshot created by
`freeze_run` is `artifacts/runs/historical-103/`. The snapshot is a recovery copy,
not a new benchmark run.

## Reproduction

From the repository root:

```bash
uv run python - <<'PY'
from pathlib import Path
from doc_harness.manifests import audit_records, audit_verdicts

records = audit_records(
    Path("artifacts/v2-runs.jsonl"), Path("artifacts/v2-predictions.json")
)
verdicts = audit_verdicts(Path("artifacts/v2-scored-103.json"))
print(records)
print(verdicts)
PY
```

The frozen inventory is written by:

```python
freeze_run(
    [
        Path("artifacts/v2-predictions.json"),
        Path("artifacts/v2-runs.jsonl"),
        Path("artifacts/v2-scored-103.json"),
        Path("artifacts/v2-scored-103.txt"),
    ],
    Path("artifacts/runs/historical-103"),
)
```

## Findings

| Check | Result |
| --- | ---: |
| Run records | 103 total; 103 `ok`; 0 failed |
| Predictions | 103; 0 malformed |
| Key coverage | Complete; 0 duplicate, missing, or unknown keys |
| Record/prediction response mismatches | 0 |
| Runtime failures | 0 known failures |
| Empty response artifacts | 0 |
| Suspected truncation | 11 responses |
| Missing termination metadata | 103 responses |
| Scored verdicts | 103 valid; 0 `judge failed:` markers |
| Semantic equivalents | 41 / 103 |

The historical records are internally reconciled, but `ok` means only that the
runner captured no exception. The records carry no tokenizer termination metadata,
so 103 outputs cannot be proven complete. Eleven have conservative truncation
signals (for example, unbalanced delimiters or an unfinished closing clause); these
are labelled *suspected truncation*, not confirmed failures. The response files are
otherwise nonempty.

The scored projection contains 103 structurally valid judge verdicts and no upstream
failure marker. It reports the historical 41/103 accuracy (39.81%) and 39.24% F1 in
`artifacts/v2-scored-103.txt`. Since the evaluator's verdicts are valid, the audit
does not rewrite them. Future evaluation uses the strict validator in
`doc_harness.evaluation`, where a transport or parse error remains an evaluation
error instead of becoming `equivalent: false`.

The freeze manifest marks code, configuration, dataset, sample, model, prompt, and
dependency identities as `unknown`: those values were not recorded by the original
run and must not be reconstructed from the current checkout. Its four copied files
retain their original bytes and SHA-256 inventory.
