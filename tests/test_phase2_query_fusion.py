from doc_harness.core.contracts import RankedPage
from doc_harness.models.retrieval import fuse_rankings


def test_reciprocal_rank_fusion_preserves_page_identity_and_tie_breaks():
    result = fuse_rankings(
        [
            [RankedPage(page_id=2, score=1.0, rank=1, stage="query"), RankedPage(page_id=1, score=0.5, rank=2, stage="query")],
            [RankedPage(page_id=1, score=1.0, rank=1, stage="query"), RankedPage(page_id=3, score=0.5, rank=2, stage="query")],
        ],
        top_k=3,
    )
    assert [item.page_id for item in result] == [1, 2, 3]
    assert [item.rank for item in result] == [1, 2, 3]


def test_reciprocal_rank_fusion_rejects_duplicate_ranks_in_one_query():
    pages = [RankedPage(page_id=1, score=1.0, rank=1, stage="query"), RankedPage(page_id=2, score=0.5, rank=1, stage="query")]
    try:
        fuse_rankings([pages], top_k=2)
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("duplicate ranks should fail")
