import pytest
from pydantic import ValidationError

from doc_harness.documents.graph import DocumentGraph, GraphEdge, GraphNode, GraphSearchState


def _node(node_id: str, *, document_id: str = "document-a", page_ids: list[int] | None = None) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        document_id=document_id,
        kind="page",
        label=node_id,
        page_ids=[] if page_ids is None else page_ids,
    )


def _graph(*, nodes: list[GraphNode], edges: list[GraphEdge]) -> DocumentGraph:
    return DocumentGraph(
        document_id="document-a",
        source_sha256="source-hash",
        extraction_version="graph-v1",
        nodes=nodes,
        edges=edges,
    )


def test_graph_contracts_are_importable():
    """Catches removal of the public graph contract entry points."""

    assert GraphSearchState().expansion_round == 0


def test_document_graph_rejects_unknown_edge_endpoint():
    """Catches edges that reference nodes outside the serialized graph."""

    with pytest.raises(ValueError, match="unknown edge endpoint"):
        _graph(
            nodes=[_node("page-1")],
            edges=[
                GraphEdge(
                    source_id="page-1",
                    relation="follows",
                    target_id="missing",
                    source_locator="page-order",
                )
            ],
        )


def test_document_graph_rejects_semantic_edge_without_provenance():
    """Catches ungrounded semantic claims entering the graph."""

    with pytest.raises(ValueError, match="provenance"):
        _graph(
            nodes=[_node("claim-1"), _node("claim-2")],
            edges=[GraphEdge(source_id="claim-1", relation="supports", target_id="claim-2")],
        )


def test_document_graph_rejects_semantic_edge_with_whitespace_only_quote():
    """Catches semantic claims with an empty quote disguised as whitespace."""

    with pytest.raises(ValueError, match="provenance"):
        _graph(
            nodes=[_node("claim-1"), _node("claim-2")],
            edges=[
                GraphEdge(
                    source_id="claim-1",
                    relation="supports",
                    target_id="claim-2",
                    page_ids=[1],
                    quote=" \t ",
                )
            ],
        )


def test_document_graph_rejects_structural_edge_without_locator():
    """Catches structural relationships that have no deterministic origin."""

    with pytest.raises(ValueError, match="source_locator"):
        _graph(
            nodes=[_node("page-1"), _node("page-2")],
            edges=[GraphEdge(source_id="page-1", relation="adjacent_to", target_id="page-2")],
        )


def test_document_graph_rejects_duplicate_node_ids():
    """Catches ambiguous node references after graph serialization."""

    with pytest.raises(ValueError, match="duplicate node ID"):
        _graph(nodes=[_node("page-1"), _node("page-1")], edges=[])


def test_document_graph_rejects_cross_document_node_provenance():
    """Catches graph nodes whose page evidence belongs to another document."""

    with pytest.raises(ValueError, match="same document"):
        _graph(nodes=[_node("page-1", document_id="document-b", page_ids=[1])], edges=[])


def test_graph_models_reject_negative_page_ids():
    """Catches invalid page provenance before it can be used for retrieval."""

    with pytest.raises(ValidationError):
        _node("page-1", page_ids=[-1])


def test_graph_search_state_rejects_negative_page_ids():
    """Catches invalid page IDs in graph retrieval state."""

    with pytest.raises(ValueError, match="page IDs"):
        GraphSearchState(seed_page_ids=[-1])
