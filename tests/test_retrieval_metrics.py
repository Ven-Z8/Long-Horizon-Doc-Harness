import pytest

from doc_harness.contracts import BenchmarkSample
from doc_harness.evaluation import retrieval_metrics


def sample(question: str, evidence_pages: list[int]):
    return BenchmarkSample(
        doc_id="doc-a",
        question=question,
        answer="answer",
        evidence_pages=evidence_pages,
    )


def test_retrieval_metrics_reports_partial_and_complete_evidence_recall():
    labelled = [sample("one", [0]), sample("two", [1, 2]), sample("empty", [])]
    selected = {
        ("doc-a", "one"): [0],
        ("doc-a", "two"): [1],
        ("doc-a", "empty"): [9],
    }

    metrics = retrieval_metrics(selected, labelled)

    assert metrics["num_samples"] == 3
    assert metrics["num_with_evidence"] == 2
    assert metrics["empty_evidence"] == 1
    assert metrics["mean_evidence_recall"] == pytest.approx(0.75)
    assert metrics["complete_evidence_recall"] == pytest.approx(0.5)
    assert metrics["hit_rate_at_k"]["4"] == pytest.approx(1.0)
    assert metrics["single_page"]["num_samples"] == 1
    assert metrics["cross_page"]["num_samples"] == 1


def test_retrieval_metrics_can_convert_explicit_one_based_labels():
    labelled = [sample("two", [2, 3])]
    selected = {("doc-a", "two"): [1, 2]}

    metrics = retrieval_metrics(selected, labelled, evidence_page_base=1)

    assert metrics["mean_evidence_recall"] == pytest.approx(1.0)
    assert metrics["complete_evidence_recall"] == pytest.approx(1.0)


def test_retrieval_metrics_normalizes_question_keys_and_counts_missing_as_misses():
    labelled = [sample("  spaced\n question ", [0])]

    metrics = retrieval_metrics({}, labelled)

    assert metrics["mean_evidence_recall"] == 0.0
    assert metrics["complete_evidence_recall"] == 0.0
