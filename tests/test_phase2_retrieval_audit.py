from doc_harness.evaluation.evaluation import compare_page_rankings


def test_equal_k_comparison_does_not_confuse_pool_size_with_reranking():
    result = compare_page_rankings([0, 1, 2, 3], [3, 2, 1, 0], {3}, ks=(2, 4))
    assert result["retrieval"]["2"]["recall"] == 0.0
    assert result["reranked"]["2"]["recall"] == 1.0
    assert result["retrieval"]["4"]["recall"] == result["reranked"]["4"]["recall"]


def test_empty_evidence_has_explicit_zero_denominator():
    result = compare_page_rankings([0, 1], [], set(), ks=(2,))
    assert result["retrieval"]["2"]["evidence_count"] == 0
    assert result["retrieval"]["2"]["recall"] is None
