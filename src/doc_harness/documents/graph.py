"""Strict, provenance-preserving contracts for document graph artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..core.contracts import StrictModel

STRUCTURAL_RELATIONS = {"contains", "child_of", "follows", "adjacent_to"}
SEMANTIC_RELATIONS = {
    "defines",
    "supports",
    "qualifies",
    "contradicts",
    "depends_on",
}


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
        known_ids = set(node_ids)
        for edge in self.edges:
            if edge.source_id not in known_ids or edge.target_id not in known_ids:
                raise ValueError("unknown edge endpoint")
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
