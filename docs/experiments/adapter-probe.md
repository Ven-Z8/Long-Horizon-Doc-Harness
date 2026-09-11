# Real checkpoint feasibility probes

These probes were run on the RTX 4090 in the workspace after the adapters were
implemented. They are smoke checks for loading and one small operation; they are
not benchmark scores.

| Adapter | Probe | Result |
| --- | --- | --- |
| Qwen3-VL-Embedding-2B | render `welcome-to-nus.pdf` at 144 DPI and embed 24 pages | passed; 2,048-dimensional finite vectors persisted and reloaded with the same checksum |
| Qwen3-VL-Reranker-2B | one question, one candidate page | passed; returned a finite score (`0.52734375`) |
| Qianfan-OCR | one 72-DPI page, 64-token generation | passed after using the documented chat-template input and image-text generation class |

The embedding checkpoint's bundled script requires `qwen-vl-utils`; the reranker
also imports SciPy. Both are in the `gpu` extra and locked in `uv.lock`. The
reranker script returns multimodal token-type IDs as a Python list with current
Transformers, while the model expects a padded tensor; the harness adapter pads
that field against the attention mask without modifying the downloaded script.

Qianfan-OCR must be loaded through `AutoModelForImageTextToText`, not generic
`AutoModel`, and its prompt must use `processor.apply_chat_template` so image
placeholder tokens match the vision features. The harness slices generated IDs
away from the prompt before storing Markdown.

The full staged benchmark remains a separate cost and latency measurement. Run a
fresh `--limit 1` pipeline before increasing the development subset.
