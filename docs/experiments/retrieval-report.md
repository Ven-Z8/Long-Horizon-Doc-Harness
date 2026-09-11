# Retrieval gate status

The retrieval metric implementation is complete and covered by offline tests in
`tests/test_retrieval_metrics.py`. It reports mean evidence-page recall,
complete-evidence coverage, hit rates at 4/8/16/20, and single-page versus
cross-page strata using labels only at evaluation time.

No benchmark retrieval score is claimed yet. The next measured run should build
the development split first, write selection manifests for the same sample keys
under embedding-only and reranked variants, and then call:

```python
from pathlib import Path
from doc_harness.batch import load_v2_samples
from doc_harness.evaluation import retrieval_metrics

samples = load_v2_samples(Path("benchmark/mmlongbench-doc-v2/data/samples.json"))
report = retrieval_metrics(selection_by_key, samples)
print(report)
```

The one-document real probe confirmed that the embedding checkpoint produces
finite 2,048-dimensional vectors and that the reranker returns a finite score.
That feasibility result does not substitute for evidence recall on the frozen
development documents.
