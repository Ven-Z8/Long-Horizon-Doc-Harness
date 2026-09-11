# Benchmark data manifest

The local benchmark assets are intentionally ignored by Git because they are large.

- `documents/`: 135 upstream MMLongBench-Doc PDFs from commit `d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384` of [mayubo2333/MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc).
- `../benchmark/mmlongbench-doc-v2/data/samples.json`: 1,071 corrected V2 questions over 134 documents.
- `../benchmark/mmlongbench-doc-v2/eval/`: V2 evaluator at commit `3aba3c6831a432ee882f763435f9ebe92ab75ed9` of [VectifyAI/MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2).

The upstream set contains one PDF that is not referenced by the V2 samples. The V2 evaluator data and correction files remain alongside the local checkout under `benchmark/mmlongbench-doc-v2/`.
