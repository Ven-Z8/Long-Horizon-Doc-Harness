from pathlib import Path

import numpy as np
import pytest

from doc_harness.contracts import Page, RankedPage, SafeQuestion
from doc_harness.retrieval import IndexIdentity, PageIndex
from doc_harness.reranking import (
    PageReranker,
    SelectionManifest,
    QwenPageReranker,
    select_page_manifest,
    select_pages,
)


def make_page(tmp_path: Path, page_id: int) -> Page:
    image = tmp_path / f"page-{page_id}.png"
    image.write_bytes(b"png")
    return Page(
        document_id="doc-a",
        page_id=page_id,
        image_path=str(image),
        width=10,
        height=10,
        render_sha256=f"hash-{page_id}",
    )


def make_index(tmp_path: Path, pages: list[Page]) -> PageIndex:
    return PageIndex(
        pages,
        np.eye(len(pages), dtype=np.float32),
        identity=IndexIdentity(
            document_id="doc-a",
            pdf_sha256="pdf",
            render_fingerprint="render",
            model_id="embed",
            model_revision="rev",
            embedding_instruction="instruction",
        ),
    )


class FakeReranker:
    def __init__(self, values):
        self.values = values
        self.seen = []

    def rank(self, question, pages):
        self.seen.append((question, pages))
        return [
            RankedPage(page_id=page_id, score=score, rank=rank, stage="rerank")
            for rank, (page_id, score) in enumerate(self.values, start=1)
        ]


def test_selection_deduplicates_candidates_stabilizes_ties_and_keeps_scores(
    tmp_path: Path,
):
    pages = [make_page(tmp_path, page_id) for page_id in range(3)]
    index = make_index(tmp_path, pages)
    question = SafeQuestion(document_id="doc-a", question="Which page?")
    reranker = FakeReranker([(2, 0.5), (0, 0.5), (2, 0.5), (1, 0.1)])

    manifest = select_page_manifest(
        question, index, reranker, candidate_k=3, selected_k=2
    )

    assert [item.page_id for item in manifest.selected] == [0, 2]
    assert [item.retrieval_score for item in manifest.selected]
    assert [item.rerank_score for item in manifest.selected] == [0.5, 0.5]
    assert [item.rank for item in select_pages(question, index, reranker, 3, 2)] == [1, 2]
    assert manifest.model_dump()["question"] == "Which page?"


def test_selection_caps_both_top_k_values_at_available_candidates(tmp_path: Path):
    pages = [make_page(tmp_path, page_id) for page_id in range(2)]
    index = make_index(tmp_path, pages)
    question = SafeQuestion(document_id="doc-a", question="Q")
    reranker = FakeReranker([(0, 0.8), (1, 0.7)])

    selected = select_pages(question, index, reranker, candidate_k=20, selected_k=6)

    assert [item.page_id for item in selected] == [0, 1]
    assert len(reranker.seen[0][1]) == 2


def test_selection_rejects_missing_assets_and_invalid_scores(tmp_path: Path):
    missing = Page(
        document_id="doc-a",
        page_id=0,
        image_path=str(tmp_path / "missing.png"),
        width=10,
        height=10,
        render_sha256="missing",
    )
    index = make_index(tmp_path, [missing])
    question = SafeQuestion(document_id="doc-a", question="Q")

    with pytest.raises(FileNotFoundError, match="page asset"):
        select_pages(question, index, FakeReranker([(0, 1.0)]), 1, 1)

    present = make_page(tmp_path, 0)
    index = make_index(tmp_path, [present])
    with pytest.raises(ValueError, match="finite"):
        select_pages(question, index, FakeReranker([(0, float("nan"))]), 1, 1)


def test_selection_rejects_bad_top_k_values(tmp_path: Path):
    page = make_page(tmp_path, 0)
    index = make_index(tmp_path, [page])
    question = SafeQuestion(document_id="doc-a", question="Q")
    reranker = FakeReranker([(0, 1.0)])
    with pytest.raises(ValueError, match="positive"):
        select_pages(question, index, reranker, candidate_k=0, selected_k=1)
    with pytest.raises(ValueError, match="positive"):
        select_pages(question, index, reranker, candidate_k=1, selected_k=0)


def test_qwen_reranker_adapter_uses_pair_scoring_backend(tmp_path: Path):
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def process(self, inputs):
            self.calls.append(inputs)
            return [0.25 for _ in inputs["documents"]]

    backend = FakeBackend()
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    reranker = QwenPageReranker(
        model_name_or_path="unused", instruction="Retrieve relevant evidence.", backend=backend
    )
    pages = [
        Page(
            document_id="doc-a",
            page_id=0,
            image_path=str(image),
            width=10,
            height=10,
            render_sha256="hash",
        )
    ]
    result = reranker.rank(
        SafeQuestion(document_id="doc-a", question="What?"), pages
    )
    assert result[0].score == pytest.approx(0.25)
    assert backend.calls[0]["query"] == {"text": "What?"}
    assert backend.calls[0]["documents"] == [{"image": str(image)}]


def test_page_reranker_is_a_protocol():
    assert hasattr(PageReranker, "rank")
