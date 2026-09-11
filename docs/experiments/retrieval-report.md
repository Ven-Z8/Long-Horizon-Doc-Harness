# Retrieval and reranking stage

This report records the offline Stage 2 contract and the measurement boundary.
The implementation keeps page IDs zero based internally, scopes exact cosine
search to a document, breaks equal scores by ascending page ID, and stores
screening and reranking scores independently in selection manifests.

No checkpoint benchmark was run in the CPU test environment. The pinned
Qwen3-VL embedding and reranking adapters are lazy: a worker loads the bundled
checkpoint script only when the real adapter is constructed. The offline suite
uses hand constructed vectors and an injected scoring backend, so it requires
neither CUDA nor downloaded model weights.

Before comparing retrieval variants, confirm the benchmark's evidence-page
numbering for the reviewed development fixture. Pass that decision explicitly
to `retrieval_metrics` with `evidence_page_base=0` or `1`; the metric helper
never guesses from labels. Empty-evidence rows are reported separately and do
not enter evidence-recall denominators.

The first real run should record model revision, embedding instruction, source
PDF hash, render settings, vector dimension, latency, and peak VRAM alongside
the index fingerprint. Reranking should be retained only when its measured
evidence coverage or downstream answer benefit justifies its added cost.
