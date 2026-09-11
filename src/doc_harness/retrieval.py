"""Document-scoped page embeddings and deterministic exact search.

The model adapter in this module deliberately has a small boundary.  It turns
``Page`` and ``SafeQuestion`` values into the input dictionaries documented by
the pinned Qwen3-VL embedding checkpoint.  Everything after that boundary is
NumPy-only and can be exercised without model weights or a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import Page, RankedPage, SafeQuestion


_INDEX_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the content hash used for PDF and rendered-page identities."""

    return hashlib.sha256(data).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return sha256_bytes(contiguous.tobytes())


def _fingerprint_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return sha256_bytes(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return _canonical_json(value)


@dataclass(frozen=True)
class IndexIdentity:
    """Inputs that determine whether an embedding cache can be reused.

    ``pdf_sha256`` identifies the source bytes, while
    ``render_fingerprint`` identifies the render policy (for example DPI and
    pixel limits).  Keeping both makes it possible to distinguish a changed
    source document from a changed rendering configuration.
    """

    document_id: str
    pdf_sha256: str = ""
    render_fingerprint: str = ""
    model_id: str = ""
    model_revision: str = ""
    embedding_instruction: str = ""

    @classmethod
    def from_inputs(
        cls,
        document_id: str,
        *,
        pdf_bytes: bytes | None = None,
        pdf_sha256: str | None = None,
        render_settings: Mapping[str, Any] | str | None = None,
        render_fingerprint: str | None = None,
        model_id: str = "",
        model_revision: str = "",
        embedding_instruction: str = "",
    ) -> "IndexIdentity":
        """Construct an identity from raw bytes or already computed hashes."""

        if pdf_sha256 is None:
            pdf_sha256 = sha256_bytes(pdf_bytes) if pdf_bytes is not None else ""
        if render_fingerprint is None:
            render_fingerprint = _fingerprint_value(render_settings)
        return cls(
            document_id=str(document_id),
            pdf_sha256=str(pdf_sha256),
            render_fingerprint=str(render_fingerprint),
            model_id=str(model_id),
            model_revision=str(model_revision),
            embedding_instruction=str(embedding_instruction),
        )

    @property
    def fingerprint(self) -> str:
        """Hash all cache-relevant model and input settings."""

        return _sha256_json(asdict(self))

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def build_index_identity(
    document_id: str,
    *,
    pdf_bytes: bytes | None = None,
    pdf_sha256: str | None = None,
    render_settings: Mapping[str, Any] | str | None = None,
    render_fingerprint: str | None = None,
    model_id: str = "",
    model_revision: str = "",
    embedding_instruction: str = "",
) -> IndexIdentity:
    """Convenience constructor for callers building an embedding cache."""

    return IndexIdentity.from_inputs(
        document_id,
        pdf_bytes=pdf_bytes,
        pdf_sha256=pdf_sha256,
        render_settings=render_settings,
        render_fingerprint=render_fingerprint,
        model_id=model_id,
        model_revision=model_revision,
        embedding_instruction=embedding_instruction,
    )


def _as_numpy(values: Any) -> np.ndarray:
    """Convert NumPy, Torch CPU, and Torch CUDA outputs without importing Torch."""

    detached = getattr(values, "detach", None)
    if callable(detached):
        values = detached()
    cpu = getattr(values, "cpu", None)
    if callable(cpu):
        values = cpu()
    numpy_method = getattr(values, "numpy", None)
    if callable(numpy_method):
        values = numpy_method()
    return np.asarray(values)


def normalize_embeddings(values: Any, *, expected_dim: int | None = None) -> np.ndarray:
    """Validate and L2-normalize a two-dimensional embedding matrix."""

    matrix = _as_numpy(values)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings cannot be empty")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("embeddings must be numeric")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("embeddings must have non-zero finite norms")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dim}, got {matrix.shape[1]}"
        )
    return (matrix / norms[:, None]).astype(np.float32, copy=False)


def _normalize_query(vector: Any, expected_dim: int) -> np.ndarray:
    values = _as_numpy(vector)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1:
        raise ValueError("query vector must be one-dimensional")
    if values.shape[0] != expected_dim:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dim}, got {values.shape[0]}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("query vector must be numeric")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("query vector must contain only finite values")
    norm = float(np.linalg.norm(values))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("query vector must have a non-zero finite norm")
    return (values / norm).astype(np.float32, copy=False)


def _check_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer")
    if int(top_k) <= 0:
        raise ValueError("top_k must be a positive integer")
    return int(top_k)


def _page_payload(page: Page) -> dict[str, Any]:
    return page.model_dump(mode="json")


def index_fingerprint(
    pages: Sequence[Page],
    *,
    identity: IndexIdentity | None = None,
    pdf_sha256: str = "",
    render_fingerprint: str = "",
    model_id: str = "",
    model_revision: str = "",
    embedding_instruction: str = "",
    embedding_dim: int | None = None,
) -> str:
    """Fingerprint pages and every setting that can change their embeddings."""

    page_payload = [_page_payload(page) for page in sorted(pages, key=lambda item: (item.document_id, item.page_id))]
    if identity is None:
        documents = sorted({page.document_id for page in pages})
        identity = IndexIdentity(
            document_id=documents[0] if len(documents) == 1 else "",
            pdf_sha256=pdf_sha256,
            render_fingerprint=render_fingerprint,
            model_id=model_id,
            model_revision=model_revision,
            embedding_instruction=embedding_instruction,
        )
    return _sha256_json(
        {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "identity": identity.as_dict(),
            "pages": page_payload,
            "embedding_dim": embedding_dim,
        }
    )


compute_index_fingerprint = index_fingerprint


@runtime_checkable
class PageEmbedder(Protocol):
    """Model-independent page/question embedding interface."""

    def encode_pages(self, pages: list[Page]) -> np.ndarray:
        ...

    def encode_questions(self, questions: list[SafeQuestion]) -> np.ndarray:
        ...


class TransformersPageEmbedder:
    """Adapter for the bundled Qwen3-VL embedding implementation.

    The import of Transformers, Torch, and ``qwen-vl-utils`` happens only when
    no backend is injected.  Tests and callers using another adapter therefore
    do not need the GPU extras installed.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        instruction: str = "Retrieve relevant evidence for the user's question.",
        model_revision: str = "",
        model_id: str | None = None,
        backend: Any | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.instruction = str(instruction)
        self.model_revision = str(model_revision)
        self.model_id = str(model_id if model_id is not None else model_name_or_path)
        self._backend = backend
        if self._backend is None:
            self._backend = self._load_backend(model_kwargs)

    def _load_backend(self, model_kwargs: Mapping[str, Any]) -> Any:
        model_path = Path(self.model_name_or_path)
        script_path = model_path / "scripts" / "qwen3_vl_embedding.py"
        if not script_path.is_file():
            raise RuntimeError(
                "the Qwen3-VL embedding adapter requires a local checkpoint with "
                "scripts/qwen3_vl_embedding.py"
            )
        module_name = "doc_harness_qwen3_vl_embedding"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load embedding adapter from {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Qwen3VLEmbedder(
            model_name_or_path=self.model_name_or_path,
            default_instruction=self.instruction,
            **dict(model_kwargs),
        )

    def _encode(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        result = self._backend.process(inputs)
        return normalize_embeddings(result)

    def encode_pages(self, pages: list[Page]) -> np.ndarray:
        if not pages:
            raise ValueError("at least one page is required")
        inputs = [
            {"image": page.image_path, "instruction": self.instruction}
            for page in pages
        ]
        result = self._encode(inputs)
        if result.shape[0] != len(pages):
            raise ValueError(
                f"embedding row count mismatch: expected {len(pages)}, got {result.shape[0]}"
            )
        return result

    def encode_questions(self, questions: list[SafeQuestion]) -> np.ndarray:
        if not questions:
            raise ValueError("at least one question is required")
        inputs = [
            {"text": question.question, "instruction": self.instruction}
            for question in questions
        ]
        result = self._encode(inputs)
        if result.shape[0] != len(questions):
            raise ValueError(
                f"embedding row count mismatch: expected {len(questions)}, got {result.shape[0]}"
            )
        return result


# A descriptive alias keeps the model-specific implementation discoverable
# while allowing callers to use the same name as the checkpoint family.
Qwen3VLPageEmbedder = TransformersPageEmbedder


class PageIndex:
    """A normalized, persistable exact-cosine index for rendered pages."""

    schema_version = _INDEX_SCHEMA_VERSION

    def __init__(
        self,
        pages: Sequence[Page],
        embeddings: Any,
        *,
        identity: IndexIdentity | None = None,
    ) -> None:
        self._pages = tuple(pages)
        if not self._pages:
            raise ValueError("at least one page is required")
        seen: set[tuple[str, int]] = set()
        for page in self._pages:
            key = (page.document_id, page.page_id)
            if key in seen:
                raise ValueError(f"duplicate page identity: {page.document_id}:{page.page_id}")
            seen.add(key)

        self._vectors = normalize_embeddings(embeddings)
        if self._vectors.shape[0] != len(self._pages):
            raise ValueError(
                "embedding row count mismatch: "
                f"expected {len(self._pages)}, got {self._vectors.shape[0]}"
            )
        documents = sorted({page.document_id for page in self._pages})
        if identity is not None and identity.document_id and identity.document_id not in documents:
            raise ValueError("index identity document_id does not match its pages")
        if identity is None:
            identity = IndexIdentity(document_id=documents[0] if len(documents) == 1 else "")
        self.identity = identity

    @property
    def pages(self) -> list[Page]:
        return list(self._pages)

    @property
    def vectors(self) -> np.ndarray:
        """Return a read-only view of normalized float32 vectors."""

        view = self._vectors.view()
        view.flags.writeable = False
        return view

    @property
    def embedding_dim(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def fingerprint(self) -> str:
        return index_fingerprint(
            self._pages,
            identity=self.identity,
            embedding_dim=self.embedding_dim,
        )

    @property
    def cache_key(self) -> str:
        return self.fingerprint

    def search(
        self,
        vector: Any,
        top_k: int,
        *,
        document_id: str | None = None,
    ) -> list[RankedPage]:
        """Return the highest cosine scores with page-ID tie breaking.

        An index containing multiple documents must be searched with an
        explicit ``document_id``.  This prevents a caller from accidentally
        using another document's pages when page IDs overlap.
        """

        limit = _check_top_k(top_k)
        documents = {page.document_id for page in self._pages}
        if document_id is None:
            if len(documents) != 1:
                raise ValueError("document_id is required for a multi-document index")
            document_id = next(iter(documents))
        if document_id not in documents:
            raise ValueError(f"unknown document_id: {document_id}")

        candidates = [
            (position, page)
            for position, page in enumerate(self._pages)
            if page.document_id == document_id
        ]
        if vector is None:
            # ``select_pages`` can be run in a separate reranking phase when
            # the screening vector was materialized by an earlier worker.  In
            # that mode this method supplies deterministic document-scoped
            # candidates; production retrieval should always pass a vector.
            scored = [(page.page_id, 0.0, page) for _position, page in candidates]
        else:
            query = _normalize_query(vector, self.embedding_dim)
            scored = [
                (page.page_id, float(np.dot(self._vectors[position], query)), page)
                for position, page in candidates
            ]
        if any(not np.isfinite(score) for _, score, _ in scored):
            raise ValueError("search produced a non-finite score")
        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = scored[: min(limit, len(scored))]
        return [
            RankedPage(page_id=page_id, score=score, rank=rank, stage="screen")
            for rank, (page_id, score, _page) in enumerate(selected, start=1)
        ]

    def save(self, path: Path) -> None:
        """Atomically persist vectors and metadata to a compressed NPZ file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.schema_version,
            "identity": self.identity.as_dict(),
            "fingerprint": self.fingerprint,
            "vectors_sha256": _array_sha256(self._vectors),
            "embedding_dim": self.embedding_dim,
            "pages": [_page_payload(page) for page in self._pages],
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                np.savez_compressed(
                    handle,
                    vectors=self._vectors,
                    metadata=np.asarray(_canonical_json(metadata)),
                )
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_identity: IndexIdentity | None = None,
        expected_fingerprint: str | None = None,
        expected_pages: Sequence[Page] | None = None,
    ) -> "PageIndex":
        """Load an index and reject stale or structurally invalid cache data."""

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                vectors = archive["vectors"]
                metadata_raw = archive["metadata"]
                if metadata_raw.ndim != 0:
                    raise ValueError("invalid index metadata")
                metadata = json.loads(str(metadata_raw.item()))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid page index cache: {path}") from exc
        if metadata.get("schema_version") != _INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported page index schema version")
        try:
            identity = IndexIdentity(**metadata["identity"])
            pages = [Page.model_validate(payload) for payload in metadata["pages"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid page index metadata") from exc
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("page index identity mismatch")
        index = cls(pages, vectors, identity=identity)
        vectors_sha256 = metadata.get("vectors_sha256")
        if vectors_sha256 is not None and vectors_sha256 != _array_sha256(index._vectors):
            raise ValueError("page index vector checksum mismatch")
        saved_fingerprint = metadata.get("fingerprint")
        if saved_fingerprint != index.fingerprint:
            raise ValueError("page index fingerprint mismatch")
        if expected_fingerprint is not None and expected_fingerprint != index.fingerprint:
            raise ValueError("page index identity mismatch")
        if expected_pages is not None:
            expected_fingerprint_from_pages = index_fingerprint(
                expected_pages,
                identity=identity,
                embedding_dim=index.embedding_dim,
            )
            if expected_fingerprint_from_pages != index.fingerprint:
                raise ValueError("page index identity mismatch")
        return index


__all__ = [
    "IndexIdentity",
    "PageEmbedder",
    "PageIndex",
    "Qwen3VLPageEmbedder",
    "TransformersPageEmbedder",
    "build_index_identity",
    "compute_index_fingerprint",
    "index_fingerprint",
    "normalize_embeddings",
    "sha256_bytes",
]
