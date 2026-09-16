import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from doc_harness.core.config import HarnessConfig
from doc_harness.documents.graph import (
    DocumentGraph,
    GraphEdge,
    GraphNode,
    GraphSearchState,
    build_document_graph,
    graph_construction_identity,
    graph_fingerprint,
    read_graph,
    write_graph,
)
from doc_harness.workflow import stages
from doc_harness.workflow.stages import build_graphs, load_graph_manifest


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


def test_document_graph_rejects_duplicate_page_node_page_ids():
    """Catches two page nodes claiming the same document page."""

    with pytest.raises(ValueError, match="duplicate page node page ID"):
        _graph(
            nodes=[
                _node("page-1", page_ids=[0]),
                _node("page-1-alias", page_ids=[0]),
            ],
            edges=[],
        )


@pytest.mark.parametrize("page_ids", [[], [0, 1]])
def test_document_graph_requires_one_page_id_per_page_node(page_ids: list[int]):
    """Catches page nodes that cannot define one unambiguous page identity."""

    with pytest.raises(ValueError, match="page node requires exactly one page ID"):
        _graph(nodes=[_node("page-1", page_ids=page_ids)], edges=[])


def test_document_graph_rejects_node_page_provenance_outside_page_nodes():
    """Catches node provenance for a page absent from the graph's page inventory."""

    with pytest.raises(ValueError, match="node page provenance references unknown page ID"):
        _graph(
            nodes=[
                _node("page-1", page_ids=[0]),
                GraphNode(
                    node_id="claim-1",
                    document_id="document-a",
                    kind="claim",
                    label="Claim",
                    page_ids=[1],
                ),
            ],
            edges=[],
        )


def test_document_graph_rejects_edge_page_provenance_outside_page_nodes():
    """Catches edge provenance for a page absent from the graph's page inventory."""

    with pytest.raises(ValueError, match="edge page provenance references unknown page ID"):
        _graph(
            nodes=[
                _node("page-1", page_ids=[0]),
                _node("page-2", page_ids=[1]),
            ],
            edges=[
                GraphEdge(
                    source_id="page-1",
                    relation="supports",
                    target_id="page-2",
                    page_ids=[2],
                    quote="Unsupported provenance.",
                )
            ],
        )


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


def test_build_document_graph_creates_deterministic_structural_and_reference_edges():
    """Catches changes to deterministic page IDs and explicit references."""

    graph = build_document_graph(
        "document-a.pdf",
        ["Introduction.", "See page 3 for the exception.", "The exception."],
        source_sha256="source-hash",
        extraction_version="graph-v1",
    )

    assert [node.node_id for node in graph.nodes] == [
        "document-a.pdf",
        "document-a.pdf:page:0",
        "document-a.pdf:page:1",
        "document-a.pdf:page:2",
    ]
    assert [(edge.source_id, edge.relation, edge.target_id) for edge in graph.edges] == [
        ("document-a.pdf", "contains", "document-a.pdf:page:0"),
        ("document-a.pdf", "contains", "document-a.pdf:page:1"),
        ("document-a.pdf", "contains", "document-a.pdf:page:2"),
        ("document-a.pdf:page:0", "follows", "document-a.pdf:page:1"),
        ("document-a.pdf:page:1", "follows", "document-a.pdf:page:2"),
        ("document-a.pdf:page:1", "refers_to", "document-a.pdf:page:2"),
    ]
    reference = graph.edges[-1]
    assert reference.page_ids == [1]
    assert reference.quote == "See page 3 for the exception."
    assert reference.source_locator == "page-text"


def test_graph_json_round_trip_rejects_changed_source_hash(tmp_path: Path):
    """Catches graph reuse after the PDF source bytes change."""

    graph = build_document_graph(
        "document-a.pdf",
        ["A single page."],
        source_sha256="source-hash",
        extraction_version="graph-v1",
    )
    path = tmp_path / "graph.json"

    write_graph(graph, path)

    assert read_graph(path) == graph
    with pytest.raises(ValueError, match="source hash"):
        read_graph(path, expected_source_sha256="changed-source-hash")


def test_build_document_graph_preserves_p_dot_references_and_ignores_out_of_range_pages():
    """Catches sentence splitting that loses abbreviated page references."""

    graph = build_document_graph(
        "document-a.pdf",
        ["Introduction.", "See p. 3 for the exception. See p. 99 for an appendix.", "Exception."],
        source_sha256="source-hash",
        extraction_version="graph-v1",
    )

    references = [edge for edge in graph.edges if edge.relation == "refers_to"]
    assert [(edge.target_id, edge.quote) for edge in references] == [
        ("document-a.pdf:page:2", "See p. 3 for the exception.")
    ]


@pytest.mark.parametrize(
    ("page_ids", "quote"),
    [([], "See page 2."), ([-1], "See page 2."), ([1, 2], "See page 2."), ([1], " \t ")],
)
def test_read_graph_rejects_malformed_refers_to_provenance(
    tmp_path: Path, page_ids: list[int], quote: str
):
    """Catches persisted references without exactly one source page and quote."""

    graph = build_document_graph(
        "document-a.pdf",
        ["See page 2.", "Target."],
        source_sha256="source-hash",
        extraction_version="graph-v1",
    )
    payload = graph.model_dump(mode="json")
    reference = next(edge for edge in payload["edges"] if edge["relation"] == "refers_to")
    reference["page_ids"] = page_ids
    reference["quote"] = quote
    path = tmp_path / "malformed-graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid graph artifact"):
        read_graph(path)


@pytest.mark.parametrize("source_id", ["document-a.pdf:page:1", "document-a.pdf"])
def test_read_graph_rejects_refers_to_with_non_source_page_node(
    tmp_path: Path, source_id: str
):
    """Catches references whose source node does not match their source-page evidence."""

    graph = build_document_graph(
        "document-a.pdf",
        ["See page 2.", "Target."],
        source_sha256="source-hash",
        extraction_version="graph-v1",
    )
    payload = graph.model_dump(mode="json")
    reference = next(edge for edge in payload["edges"] if edge["relation"] == "refers_to")
    reference["source_id"] = source_id
    path = tmp_path / "mismatched-reference-source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid graph artifact"):
        read_graph(path)


def test_build_graphs_writes_and_validates_document_scoped_manifest(tmp_path: Path):
    """Catches graph artifacts that omit source, parser, or configuration identity."""

    fitz = pytest.importorskip("pymupdf")
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    pdf_path = documents_dir / "document-a.pdf"
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "See page 2 for the exception.")
        document.new_page().insert_text((72, 72), "The exception.")
        document.save(pdf_path)
    finally:
        document.close()

    class _GraphConfig:
        extraction_version = "graph-v1"

    class _Config:
        graph = _GraphConfig()

        @staticmethod
        def effective_hash() -> str:
            return "config-hash"

    manifest_path = build_graphs(_Config(), documents_dir, tmp_path / "graphs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graphs = load_graph_manifest(manifest_path)

    assert manifest["config_hash"] == "config-hash"
    assert manifest["construction_identity"]
    assert manifest["entries"][0]["page_count"] == 2
    assert graphs["document-a.pdf"].extraction_version == "graph-v1"
    assert graphs["document-a.pdf"].edges[-1].relation == "refers_to"


def test_load_graph_manifest_rejects_extraction_identity_mismatch(tmp_path: Path):
    """Catches stale graph metadata whose manifest disagrees with graph bytes."""

    graph = build_document_graph("document-a.pdf", ["One page."], "source-hash", "graph-v1")
    graph_path = tmp_path / "document-a.pdf.graph.json"
    write_graph(graph, graph_path)
    manifest_path = tmp_path / "graph-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "graph",
                "config_hash": "config-hash",
                "construction_identity": graph_construction_identity("graph-v1"),
                "entries": [
                    {
                        "document_id": "document-a.pdf",
                        "source_sha256": "source-hash",
                        "schema_version": 1,
                        "extraction_version": "changed-version",
                        "graph_fingerprint": "unused",
                        "page_count": 1,
                        "graph_path": str(graph_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extraction"):
        load_graph_manifest(manifest_path)


def test_graph_manifest_rejects_construction_identity_mismatch_but_allows_traversal_ablation(
    tmp_path: Path,
):
    """Catches stale construction settings without coupling artifacts to traversal knobs."""

    graph = build_document_graph("document-a.pdf", ["One page."], "source-hash", "graph-v1")
    graph_path = tmp_path / "document-a.pdf.graph.json"
    write_graph(graph, graph_path)
    manifest_path = tmp_path / "graph-manifest.json"
    payload = {
        "schema_version": 1,
        "stage": "graph",
        "config_hash": "full-build-config-hash",
        "construction_identity": graph_construction_identity("graph-v1"),
        "entries": [
            {
                "document_id": graph.document_id,
                "source_sha256": graph.source_sha256,
                "schema_version": graph.schema_version,
                "extraction_version": graph.extraction_version,
                "graph_fingerprint": graph_fingerprint(graph),
                "page_count": 1,
                "graph_path": str(graph_path),
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_config = HarnessConfig.model_validate(
        {
            "model": {"model_id": "m", "revision": "r"},
            "graph": {
                "enabled": True,
                "max_hops": 4,
                "max_candidates": 1,
                "require_connection": True,
                "allowed_relations": ["refers_to"],
                "extraction_version": "graph-v1",
            },
        }
    )

    graphs, _fingerprint, failure = stages._graph_manifest_for_enabled_run(
        runtime_config, manifest_path
    )

    assert failure is None
    assert graphs == {"document-a.pdf": graph}

    payload["construction_identity"] = "wrong-construction-identity"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _graphs, _fingerprint, failure = stages._graph_manifest_for_enabled_run(
        runtime_config, manifest_path
    )

    assert failure is not None
    assert "construction identity" in failure
