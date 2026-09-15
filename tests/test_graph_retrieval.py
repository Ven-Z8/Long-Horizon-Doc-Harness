import pytest

from doc_harness.contracts import RankedPage
from doc_harness.documents.graph import DocumentGraph, GraphEdge, GraphNode
from doc_harness.documents.graph_retrieval import (
    expand_graph_candidates,
    select_connected_pages,
)


def _graph():
    nodes = [
        GraphNode(
            node_id=f"document-a.pdf:page:{page_id}",
            document_id="document-a.pdf",
            kind="page",
            label=f"Page {page_id + 1}",
            page_ids=[page_id],
        )
        for page_id in range(4)
    ]
    return DocumentGraph(
        document_id="document-a.pdf",
        source_sha256="source-hash",
        extraction_version="graph-v1",
        nodes=nodes,
        edges=[
            GraphEdge(
                source_id="document-a.pdf:page:0",
                relation="refers_to",
                target_id="document-a.pdf:page:2",
                page_ids=[0],
                quote="See page 3.",
            ),
            GraphEdge(
                source_id="document-a.pdf:page:2",
                relation="follows",
                target_id="document-a.pdf:page:3",
                source_locator="page-order",
            ),
        ],
    )


def test_expansion_is_seed_first_breadth_first_and_bounded():
    """Catches traversal that loses seed order or walks past configured caps."""

    assert expand_graph_candidates(
        [0], _graph(), max_hops=2, max_candidates=4,
        allowed_relations=["refers_to", "follows"],
    ) == [0, 2, 3]
    assert expand_graph_candidates(
        [0], _graph(), max_hops=1, max_candidates=2,
        allowed_relations=["refers_to", "follows"],
    ) == [0, 2]


def test_expansion_filters_relations_and_rejects_invalid_caps():
    """Catches structural edges slipping through a restricted graph route."""

    assert expand_graph_candidates(
        [0], _graph(), max_hops=2, max_candidates=4,
        allowed_relations=["refers_to"],
    ) == [0, 2]
    with pytest.raises(ValueError, match="max_hops"):
        expand_graph_candidates([], _graph(), max_hops=-1, max_candidates=1, allowed_relations=[])
    with pytest.raises(ValueError, match="max_candidates"):
        expand_graph_candidates([], _graph(), max_hops=0, max_candidates=0, allowed_relations=[])


def test_connected_selection_prefers_adjacent_pages_then_fills_by_score():
    """Catches selection that discards graph connectivity after reranking."""

    ranked = [
        RankedPage(page_id=0, score=0.9, rank=1, stage="rerank"),
        RankedPage(page_id=1, score=0.8, rank=2, stage="rerank"),
        RankedPage(page_id=2, score=0.7, rank=3, stage="rerank"),
        RankedPage(page_id=3, score=0.6, rank=4, stage="rerank"),
    ]

    connected = select_connected_pages(ranked, _graph(), selected_k=3, require_connection=True)
    unconnected = select_connected_pages(ranked, _graph(), selected_k=3, require_connection=False)

    assert [page.page_id for page in connected] == [0, 2, 3]
    assert connected[0] is ranked[0]
    assert [page.page_id for page in unconnected] == [0, 1, 2]
