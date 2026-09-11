from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from doc_harness.contracts import Page, SafeQuestion
from doc_harness.retrieval import (
    IndexIdentity,
    PageIndex,
    PageEmbedder,
    TransformersPageEmbedder,
)


def page(document_id: str, page_id: int) -> Page:
    return Page(
        document_id=document_id,
        page_id=page_id,
        image_path=f"{document_id}-{page_id}.png",
        width=100,
        height=100,
        render_sha256=f"render-{document_id}-{page_id}",
    )


def identity(document_id: str = "doc-a") -> IndexIdentity:
    return IndexIdentity(
        document_id=document_id,
        pdf_sha256="pdf-hash",
        render_fingerprint="render-settings",
        model_id="embedder",
        model_revision="revision-a",
        embedding_instruction="Retrieve evidence relevant to the question.",
    )


def test_exact_cosine_search_orders_nearest_pages_and_stable_ties():
    pages = [page("doc-a", page_id) for page_id in [2, 0, 1]]
    vectors = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    index = PageIndex(pages, vectors, identity=identity())

    results = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=3)

    assert [item.page_id for item in results] == [0, 1, 2]
    assert [item.rank for item in results] == [1, 2, 3]
    assert [item.stage for item in results] == ["screen"] * 3
    assert results[0].score == pytest.approx(results[1].score)


def test_mixed_document_index_requires_document_scope():
    pages = [page("doc-a", 0), page("doc-b", 0)]
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index = PageIndex(pages, vectors, identity=None)

    with pytest.raises(ValueError, match="document_id"):
        index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)

    results = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32), top_k=2, document_id="doc-b"
    )
    assert [item.page_id for item in results] == [0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pdf_sha256", "different-pdf"),
        ("render_fingerprint", "different-render"),
        ("model_revision", "revision-b"),
        ("embedding_instruction", "different instruction"),
    ],
)
def test_changed_identity_invalidates_saved_index(
    tmp_path: Path, field: str, value: str
):
    index = PageIndex(
        [page("doc-a", 0)], np.asarray([[1.0, 0.0]], dtype=np.float32), identity=identity()
    )
    cache_path = tmp_path / "pages.npz"
    index.save(cache_path)

    with pytest.raises(ValueError, match="identity"):
        PageIndex.load(cache_path, expected_identity=replace(identity(), **{field: value}))


def test_index_rejects_nonfinite_and_dimension_mismatch_vectors():
    pages = [page("doc-a", 0)]
    with pytest.raises(ValueError, match="finite"):
        PageIndex(pages, np.asarray([[np.nan, 0.0]], dtype=np.float32), identity=identity())

    index = PageIndex(
        pages, np.asarray([[1.0, 0.0]], dtype=np.float32), identity=identity()
    )
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)


def test_transformers_embedder_uses_documented_inputs_and_normalizes_output():
    class FakeModel:
        def __init__(self):
            self.inputs = []

        def process(self, inputs):
            self.inputs.append(inputs)
            return np.asarray([[3.0, 4.0] for _ in inputs], dtype=np.float32)

    backend = FakeModel()
    embedder = TransformersPageEmbedder(
        model_name_or_path="unused", instruction="Find relevant pages.", backend=backend
    )
    encoded_pages = embedder.encode_pages([page("doc-a", 2)])
    encoded_questions = embedder.encode_questions(
        [SafeQuestion(document_id="doc-a", question="What matters?")]
    )

    assert encoded_pages.dtype == np.float32
    assert encoded_pages.tolist() == [[pytest.approx(0.6), pytest.approx(0.8)]]
    assert encoded_questions.tolist() == [[pytest.approx(0.6), pytest.approx(0.8)]]
    assert backend.inputs[0] == [
        {"image": "doc-a-2.png", "instruction": "Find relevant pages."}
    ]
    assert backend.inputs[1] == [
        {"text": "What matters?", "instruction": "Find relevant pages."}
    ]


def test_page_embedder_is_a_runtime_protocol():
    assert isinstance(TransformersPageEmbedder, type)
    assert hasattr(PageEmbedder, "encode_pages")
