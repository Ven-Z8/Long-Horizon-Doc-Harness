"""Strict, provenance-preserving contracts for document graph artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field, model_validator

from ..core.contracts import StrictModel

STRUCTURAL_RELATIONS = {"contains", "child_of", "follows", "adjacent_to"}
SEMANTIC_RELATIONS = {
    "refers_to",
    "defines",
    "supports",
    "qualifies",
    "contradicts",
    "depends_on",
}

_PAGE_REFERENCE = re.compile(r"(?:page|p\.)\s+(\d+)", re.IGNORECASE)


class GraphNode(StrictModel):
    node_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    kind: Literal["document", "section", "page", "block", "claim", "entity"]
    label: str = Field(min_length=1)
    page_ids: list[int] = Field(default_factory=list)
    source_quote: str | None = None
    source_locator: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_page_ids(self) -> "GraphNode":
        if any(page_id < 0 for page_id in self.page_ids):
            raise ValueError("page IDs must be greater than or equal to 0")
        return self


class GraphEdge(StrictModel):
    source_id: str = Field(min_length=1)
    relation: Literal[
        "contains",
        "child_of",
        "follows",
        "adjacent_to",
        "refers_to",
        "defines",
        "supports",
        "qualifies",
        "contradicts",
        "depends_on",
    ]
    target_id: str = Field(min_length=1)
    page_ids: list[int] = Field(default_factory=list)
    quote: str | None = None
    source_locator: str | None = None
    condition: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> "GraphEdge":
        if any(page_id < 0 for page_id in self.page_ids):
            raise ValueError("page IDs must be greater than or equal to 0")
        if self.relation in STRUCTURAL_RELATIONS and not (self.source_locator or "").strip():
            raise ValueError("structural edge requires source_locator provenance")
        if self.relation in SEMANTIC_RELATIONS and (
            not self.page_ids or not (self.quote or "").strip()
        ):
            raise ValueError("semantic edge requires page_ids and quote provenance")
        if self.relation == "refers_to" and len(self.page_ids) != 1:
            raise ValueError("refers_to edge requires exactly one source page ID")
        return self


class DocumentGraph(StrictModel):
    schema_version: Literal[1] = 1
    document_id: str = Field(min_length=1)
    source_sha256: str = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_references(self) -> "DocumentGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node ID")
        if any(node.document_id != self.document_id for node in self.nodes):
            raise ValueError("node page provenance must use the same document")
        nodes_by_id = {node.node_id: node for node in self.nodes}
        known_ids = set(nodes_by_id)
        for edge in self.edges:
            if edge.source_id not in known_ids or edge.target_id not in known_ids:
                raise ValueError("unknown edge endpoint")
            if edge.relation == "refers_to":
                source_node = nodes_by_id[edge.source_id]
                if source_node.kind != "page" or source_node.page_ids != edge.page_ids:
                    raise ValueError(
                        "refers_to source must be a page node matching its source page ID"
                    )
        return self


class GraphSearchState(StrictModel):
    seed_page_ids: list[int] = Field(default_factory=list)
    candidate_page_ids: list[int] = Field(default_factory=list)
    visited_node_ids: list[str] = Field(default_factory=list)
    active_node_id: str | None = None
    missing_evidence: list[str] = Field(default_factory=list)
    expansion_round: int = Field(default=0, ge=0)
    terminal_reason: str | None = None

    @model_validator(mode="after")
    def validate_page_ids(self) -> "GraphSearchState":
        page_ids = self.seed_page_ids + self.candidate_page_ids
        if any(page_id < 0 for page_id in page_ids):
            raise ValueError("page IDs must be greater than or equal to 0")
        return self


def build_document_graph(
    document_id: str,
    page_texts: Sequence[str],
    source_sha256: str,
    extraction_version: str,
) -> DocumentGraph:
    """Build a question-independent structural graph from extracted page text."""

    document_node_id = document_id
    nodes = [
        GraphNode(
            node_id=document_node_id,
            document_id=document_id,
            kind="document",
            label=document_id,
        )
    ]
    edges: list[GraphEdge] = []
    normalized_texts: list[str] = []
    for page_id, text in enumerate(page_texts):
        if not isinstance(text, str):
            raise TypeError("page_texts must contain strings")
        normalized_texts.append(text)
        page_node_id = _page_node_id(document_id, page_id)
        nodes.append(
            GraphNode(
                node_id=page_node_id,
                document_id=document_id,
                kind="page",
                label=f"Page {page_id + 1}",
                page_ids=[page_id],
                source_locator="page-text",
            )
        )
        edges.append(
            GraphEdge(
                source_id=document_node_id,
                relation="contains",
                target_id=page_node_id,
                page_ids=[page_id],
                source_locator="document-page-order",
            )
        )

    for page_id in range(len(normalized_texts) - 1):
        edges.append(
            GraphEdge(
                source_id=_page_node_id(document_id, page_id),
                relation="follows",
                target_id=_page_node_id(document_id, page_id + 1),
                page_ids=[page_id, page_id + 1],
                source_locator="page-order",
            )
        )

    for page_id, text in enumerate(normalized_texts):
        for match in _PAGE_REFERENCE.finditer(text):
            target_page_id = int(match.group(1)) - 1
            if 0 <= target_page_id < len(normalized_texts):
                edges.append(
                    GraphEdge(
                        source_id=_page_node_id(document_id, page_id),
                        relation="refers_to",
                        target_id=_page_node_id(document_id, target_page_id),
                        page_ids=[page_id],
                        quote=_enclosing_sentence(text, match.start(), match.end()),
                        source_locator="page-text",
                    )
                )

    nodes.sort(key=lambda node: (node.kind != "document", node.page_ids, node.node_id))
    relation_order = {"contains": 0, "follows": 1, "refers_to": 2}
    edges.sort(
        key=lambda edge: (
            relation_order.get(edge.relation, 99),
            edge.source_id,
            edge.target_id,
            edge.page_ids,
            edge.quote or "",
        )
    )
    return DocumentGraph(
        document_id=document_id,
        source_sha256=source_sha256,
        extraction_version=extraction_version,
        nodes=nodes,
        edges=edges,
    )


def graph_fingerprint(graph: DocumentGraph) -> str:
    """Return the stable identity hash for a validated graph artifact."""

    encoded = json.dumps(
        graph.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_graph(graph: DocumentGraph, path: Path) -> None:
    """Atomically write a validated graph as deterministic JSON."""

    payload = graph.model_dump(mode="json")
    _write_json(Path(path), payload)


def read_graph(path: Path, expected_source_sha256: str | None = None) -> DocumentGraph:
    """Read a graph and reject artifacts from a different source document."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = DocumentGraph.model_validate(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid graph artifact: {path}") from exc
    if expected_source_sha256 is not None and graph.source_sha256 != expected_source_sha256:
        raise ValueError("graph source hash does not match expected source hash")
    return graph


def _page_node_id(document_id: str, page_id: int) -> str:
    return f"{document_id}:page:{page_id}"


def _enclosing_sentence(text: str, match_start: int, match_end: int) -> str:
    """Return the source sentence around a match without splitting ``p. 3``."""

    start = max(text.rfind(delimiter, 0, match_start) for delimiter in ".!?") + 1
    end_candidates = [
        position
        for delimiter in ".!?"
        if (position := text.find(delimiter, match_end)) >= 0
    ]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "DocumentGraph",
    "GraphEdge",
    "GraphNode",
    "GraphSearchState",
    "build_document_graph",
    "graph_fingerprint",
    "read_graph",
    "write_graph",
]
