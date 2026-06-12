"""Parquet I/O for AssemblyGraph artifacts.

Artifact lifecycle composition (callers compose; no orchestrator):

    recovery        ->  save_graph + save_visits
    paths.compute   ->  save_assembly_indices
    embedding pass  ->  save_embedding_indices
    inference       ->  load_graph + (optional) load_*

Each ``save_*`` / ``load_*`` operates on the artifact directory ``path``.
``save_graph`` creates the directory; subsequent ``save_*`` calls add
optional pieces alongside. Loads return ``None`` (or ``{}`` for FAISS
indices) for missing optional pieces; required pieces (graph) raise
``ArtifactNotFoundError``.

Atomic-write protocol:

- ``save_graph``: write to ``<path>.tmp.<uuid>/`` then rename to
  ``<path>``. Whole-directory atomicity — partial state would break
  the structural invariants (graph is meaningless without all of
  metadata + nodes + edges).
- ``save_embedding_indices``: write to ``faiss_indices.tmp.<uuid>/``
  then rename to ``faiss_indices/``. Whole-directory atomicity — the
  manifest cross-references all per-timestep files.
- ``save_visits`` and ``save_assembly_indices``: per-file atomic write
  (``<file>.tmp.<uuid>`` -> ``<file>``). Adding a single optional file
  to an existing artifact is a valid lifecycle event, so per-file
  atomicity is sufficient.

Schema versioning:

- Graph artifact: ``metadata.json`` carries ``schema_version`` (currently
  1; see ``bsig.core.graph.SCHEMA_VERSION``). FAISS manifest carries its
  own ``schema_version`` (currently 1; see ``MANIFEST_SCHEMA_VERSION``).
  Mismatched versions raise ``SchemaVersionError`` — 0.x performs no
  migrations.

Variable-length trajectories: per-timestep FAISS indices may have fewer
entries at higher timesteps. Consumers must inspect the manifest's
``timesteps`` map (timestep -> visit count) before querying.

``visits.parquet`` PK is ``visit_seq: int64``, sequential from zero.
The PK is **not** stable across re-saves; consumers that rebuild the
visits table must also rebuild any FAISS indices that referenced the
old PKs (FAISS internal IDs equal ``visit_seq``).
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from bsig.core.graph import (
    SCHEMA_VERSION,
    AssemblyGraph,
    AssemblyGraphBuilder,
    EdgeClass,
)
from bsig.core.signature import SignatureWeights

if TYPE_CHECKING:
    import faiss


MANIFEST_SCHEMA_VERSION = 1
SIGNATURE_METADATA_SCHEMA_VERSION = 1

_REQUIRED_SIGNATURE_SCORE_COLUMNS = (
    "trajectory_id",
    "entropy_plateau",
    "voi_flatness",
    "distance_from_trajectory",
    "composite",
)

_REQUIRED_VISIT_COLUMNS = (
    "visit_seq",
    "trajectory_id",
    "timestep",
    "node_id",
    "embedding",
)
_REQUIRED_ASSEMBLY_INDEX_COLUMNS = ("node_id", "assembly_index")


# ---- Errors ----


class PersistenceError(Exception):
    """Base class for persistence-related errors."""


class ArtifactNotFoundError(PersistenceError):
    """Raised when an artifact directory or required file is missing."""


class ArtifactExistsError(PersistenceError):
    """Raised when ``overwrite=False`` blocks a write to an existing target."""


class SchemaVersionError(PersistenceError):
    """Raised when a loaded artifact's schema version does not match."""

    def __init__(self, *, expected: int, actual: Any, kind: str) -> None:
        super().__init__(
            f"{kind} schema_version mismatch: expected {expected}, got {actual!r}"
        )
        self.expected = expected
        self.actual = actual
        self.kind = kind


class FaissNotInstalledError(PersistenceError):
    """Raised when FAISS is required but ``faiss-cpu`` is not installed."""


# ---- Atomic-write helpers ----


@contextmanager
def _atomic_write_dir(target: Path, *, overwrite: bool) -> Iterator[Path]:
    """Yield a temp directory; on success rename to target.

    If ``target`` exists and ``overwrite`` is True, the existing directory
    is moved aside (``<target>.old.<uuid>``) before the rename and removed
    after. If the rename fails, the previous directory is restored.
    """
    target = Path(target)
    if target.exists() and not overwrite:
        raise ArtifactExistsError(
            f"Target already exists: {target} (pass overwrite=True to replace)"
        )
    if not target.parent.exists():
        raise ArtifactNotFoundError(
            f"Parent directory does not exist: {target.parent}"
        )

    tmp = target.parent / f"{target.name}.tmp.{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=False, exist_ok=False)

    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    backup: Path | None = None
    if target.exists():
        backup = target.parent / f"{target.name}.old.{uuid.uuid4().hex[:8]}"
        os.rename(target, backup)
    try:
        os.rename(tmp, target)
    except OSError:
        if backup is not None and backup.exists():
            os.rename(backup, target)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _atomic_write_file(
    target: Path,
    write_fn: Callable[[Path], None],
    *,
    overwrite: bool,
) -> None:
    """Write to ``<target>.tmp.<uuid>`` via ``write_fn``, rename to target."""
    target = Path(target)
    if target.exists() and not overwrite:
        raise ArtifactExistsError(
            f"Target already exists: {target} (pass overwrite=True to replace)"
        )
    if not target.parent.exists():
        raise ArtifactNotFoundError(
            f"Parent directory does not exist: {target.parent}"
        )

    tmp = target.parent / f"{target.name}.tmp.{uuid.uuid4().hex[:8]}"
    try:
        write_fn(tmp)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    try:
        os.rename(tmp, target)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _import_faiss() -> Any:
    try:
        import faiss  # noqa: PLC0415
    except ImportError as exc:
        raise FaissNotInstalledError(
            "faiss-cpu is not installed; install with: "
            "uv pip install -e '.[faiss]'"
        ) from exc
    return faiss


# ---- save_graph / load_graph ----


def save_graph(
    graph: AssemblyGraph, path: Path, *, overwrite: bool = False
) -> None:
    """Write the graph artifact (metadata + nodes + edges) at ``path``.

    Creates ``path`` as an artifact directory atomically. Subsequent
    ``save_visits`` / ``save_embedding_indices`` / ``save_assembly_indices``
    add optional pieces alongside.
    """
    path = Path(path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "edge_class_encoding": {
            str(int(cls)): cls.name for cls in EdgeClass
        },
        "graph_metadata": dict(graph.metadata),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    nodes_df = pd.DataFrame(
        {
            "node_id": list(graph._nodes.index),
            "visit_count": graph._nodes["visit_count"].to_numpy(),
            "is_terminal": graph._nodes["is_terminal"].to_numpy(),
        }
    )

    with _atomic_write_dir(path, overwrite=overwrite) as tmp:
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2))
        nodes_df.to_parquet(tmp / "nodes.parquet", index=False)
        graph._edges.to_parquet(tmp / "edges.parquet", index=False)


def load_graph(path: Path) -> AssemblyGraph:
    """Load the graph artifact at ``path``.

    Raises ``ArtifactNotFoundError`` if ``path`` does not exist or is
    missing any of metadata.json, nodes.parquet, edges.parquet.
    Raises ``SchemaVersionError`` on version mismatch.
    """
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise ArtifactNotFoundError(f"Artifact directory not found: {path}")

    metadata_path = path / "metadata.json"
    nodes_path = path / "nodes.parquet"
    edges_path = path / "edges.parquet"
    for required in (metadata_path, nodes_path, edges_path):
        if not required.exists():
            raise ArtifactNotFoundError(f"Required file missing: {required}")

    metadata = json.loads(metadata_path.read_text())
    actual_version = metadata.get("schema_version")
    if actual_version != SCHEMA_VERSION:
        raise SchemaVersionError(
            expected=SCHEMA_VERSION, actual=actual_version, kind="graph"
        )

    nodes_df = pd.read_parquet(nodes_path).set_index("node_id")
    edges_df = pd.read_parquet(edges_path)

    nx_graph: nx.MultiDiGraph[str] = AssemblyGraphBuilder._build_nx(
        nodes_df, edges_df
    )
    nx.freeze(nx_graph)

    return AssemblyGraph(
        _nodes=nodes_df,
        _edges=edges_df,
        _nx=nx_graph,
        _metadata=metadata.get("graph_metadata", {}),
    )


# ---- save_visits / load_visits ----


def save_visits(
    visits: pd.DataFrame, path: Path, *, overwrite: bool = False
) -> None:
    """Write ``visits.parquet`` to the artifact directory at ``path``.

    Validates required columns. ``visit_seq`` must be unique; consumers
    are responsible for sequential numbering from zero (the contract that
    lets FAISS internal IDs equal ``visit_seq`` directly).
    """
    path = Path(path)
    if not path.is_dir():
        raise ArtifactNotFoundError(f"Artifact directory not found: {path}")

    missing = set(_REQUIRED_VISIT_COLUMNS) - set(visits.columns)
    if missing:
        raise ValueError(f"visits is missing required columns: {sorted(missing)}")
    if visits["visit_seq"].duplicated().any():
        raise ValueError("visits.visit_seq must be unique")

    target = path / "visits.parquet"
    _atomic_write_file(
        target,
        lambda p: visits.to_parquet(p, index=False),
        overwrite=overwrite,
    )


def load_visits(path: Path) -> pd.DataFrame | None:
    """Load ``visits.parquet`` from the artifact at ``path``.

    Returns ``None`` if the file does not exist (visits are optional).
    """
    path = Path(path)
    target = path / "visits.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


# ---- save_embedding_indices / load_embedding_indices ----


def save_embedding_indices(
    indices: Mapping[int, "faiss.Index"],
    path: Path,
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write per-timestep FAISS indices and manifest under ``faiss_indices/``.

    ``manifest`` carries embedding-model metadata; this function adds
    ``schema_version`` and ``timesteps`` automatically. Caller must
    provide at least: ``embedding_model``, ``dimension``,
    ``normalization``, ``index_type``. Optional: ``embedding_revision``.
    """
    faiss = _import_faiss()
    path = Path(path)
    if not path.is_dir():
        raise ArtifactNotFoundError(f"Artifact directory not found: {path}")

    timesteps = {str(t): int(idx.ntotal) for t, idx in indices.items()}
    full_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **dict(manifest),
        "timesteps": timesteps,
    }

    target = path / "faiss_indices"
    with _atomic_write_dir(target, overwrite=overwrite) as tmp:
        (tmp / "manifest.json").write_text(json.dumps(full_manifest, indent=2))
        for timestep, index in indices.items():
            faiss.write_index(index, str(tmp / f"timestep_{timestep}.faiss"))


def build_faiss_indices_from_visits(
    visits: pd.DataFrame,
    *,
    dimension: int | None = None,
    embedding_model: str = "unknown",
    embedding_revision: str | None = None,
    normalization: str = "l2",
    index_type: str = "IndexFlatIP",
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Build per-timestep FAISS indices from a visits DataFrame.

    Returns ``(indices_by_timestep, manifest)`` suitable for passing
    directly to ``save_embedding_indices``. Indices are
    ``IndexFlatIP`` over L2-normalized embeddings (cosine via inner
    product, per stage 2.4's ``distance_from_trajectory`` precondition).

    Per-timestep auto-assigned IDs (0..N-1 within each timestep). The
    visits table's ``visit_seq`` global ID is recorded in the manifest's
    timestep counts but not used as the FAISS internal ID — diagnostic
    "which trajectory did this neighbor come from?" lookups would
    require ``IndexIDMap`` wrapping; not needed for
    ``distance_from_trajectory`` and skipped for 0.1 simplicity.
    """
    faiss = _import_faiss()

    if visits.empty:
        manifest = {
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "dimension": dimension or 0,
            "normalization": normalization,
            "index_type": index_type,
        }
        return {}, manifest

    # Determine dimension from first embedding if not provided
    if dimension is None:
        first_emb = visits.iloc[0]["embedding"]
        if first_emb is None:
            raise ValueError(
                "Cannot infer dimension: first visit has embedding=None"
            )
        dimension = len(first_emb)

    indices: dict[int, Any] = {}
    for timestep, group in visits.groupby("timestep"):
        embeddings = np.stack(
            [np.asarray(e, dtype=np.float32) for e in group["embedding"]]
        )
        idx = faiss.IndexFlatIP(int(dimension))
        idx.add(embeddings)
        # pandas groupby key is broad-typed under pandas-stubs; we know
        # it's an int because the column dtype is int32.
        indices[int(timestep)] = idx  # type: ignore[arg-type]

    manifest = {
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "dimension": int(dimension),
        "normalization": normalization,
        "index_type": index_type,
    }
    return indices, manifest


def load_embedding_indices(path: Path) -> dict[int, "faiss.Index"]:
    """Load per-timestep FAISS indices from ``faiss_indices/``.

    Returns ``{}`` if no faiss_indices directory exists. Raises
    ``FaissNotInstalledError`` if faiss-cpu is missing.
    Raises ``SchemaVersionError`` on manifest version mismatch.
    Raises ``ArtifactNotFoundError`` if the manifest references a
    timestep file that does not exist on disk.
    """
    faiss = _import_faiss()
    path = Path(path)
    indices_dir = path / "faiss_indices"
    if not indices_dir.is_dir():
        return {}

    manifest_path = indices_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    manifest = json.loads(manifest_path.read_text())
    actual_version = manifest.get("schema_version")
    if actual_version != MANIFEST_SCHEMA_VERSION:
        raise SchemaVersionError(
            expected=MANIFEST_SCHEMA_VERSION,
            actual=actual_version,
            kind="faiss_manifest",
        )

    indices: dict[int, Any] = {}
    for ts_str in manifest.get("timesteps", {}):
        timestep = int(ts_str)
        index_path = indices_dir / f"timestep_{timestep}.faiss"
        if not index_path.exists():
            raise ArtifactNotFoundError(
                f"Manifest references missing index file: {index_path}"
            )
        indices[timestep] = faiss.read_index(str(index_path))
    return indices


# ---- save_assembly_indices / load_assembly_indices ----


def save_assembly_indices(
    indices: pd.DataFrame, path: Path, *, overwrite: bool = False
) -> None:
    """Write ``assembly_indices.parquet`` to the artifact at ``path``."""
    path = Path(path)
    if not path.is_dir():
        raise ArtifactNotFoundError(f"Artifact directory not found: {path}")

    missing = set(_REQUIRED_ASSEMBLY_INDEX_COLUMNS) - set(indices.columns)
    if missing:
        raise ValueError(
            f"assembly_indices is missing required columns: {sorted(missing)}"
        )

    target = path / "assembly_indices.parquet"
    _atomic_write_file(
        target,
        lambda p: indices.to_parquet(p, index=False),
        overwrite=overwrite,
    )


def load_assembly_indices(path: Path) -> pd.DataFrame | None:
    """Load ``assembly_indices.parquet`` from the artifact at ``path``.

    Returns ``None`` if the file does not exist.
    """
    path = Path(path)
    target = path / "assembly_indices.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


# ---- save_signature_scores / load_signature_scores / load_signature_weights ----


def save_signature_scores(
    scores: pd.DataFrame,
    weights: SignatureWeights,
    path: Path,
    *,
    overwrite: bool = False,
    graph_artifact_path: Path | None = None,
    prompt_versions: Mapping[str, int] | None = None,
) -> None:
    """Write ``signature_scores.parquet`` and ``signature_metadata.json``
    to the artifact directory at ``path``.

    Validates that ``scores`` carries the expected columns. The
    metadata records weights, normalization choice, optional reference
    to the source graph artifact (best-effort, for documentation —
    strict version-pinning between graph and signature artifacts is
    the caller's responsibility).

    ``prompt_versions`` (optional) records the version integer of each
    prompt template used by the run that produced these scores
    (typically from ``bsig.medqa.load_all_versions()``). Version 0
    indicates a placeholder template; downstream analysis should treat
    placeholder runs as not-for-headline-metrics.
    """
    import dataclasses

    path = Path(path)
    if not path.is_dir():
        raise ArtifactNotFoundError(f"Artifact directory not found: {path}")

    missing = set(_REQUIRED_SIGNATURE_SCORE_COLUMNS) - set(scores.columns)
    if missing:
        raise ValueError(
            f"signature scores missing required columns: {sorted(missing)}"
        )

    metadata: dict[str, Any] = {
        "schema_version": SIGNATURE_METADATA_SCHEMA_VERSION,
        "weights": dataclasses.asdict(weights),
        "normalization": "rank_percentile",
        "graph_artifact_path": (
            str(graph_artifact_path) if graph_artifact_path else None
        ),
        "n_trajectories": len(scores),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if prompt_versions is not None:
        metadata["prompt_versions"] = dict(prompt_versions)

    scores_target = path / "signature_scores.parquet"
    metadata_target = path / "signature_metadata.json"

    _atomic_write_file(
        scores_target,
        lambda p: scores.to_parquet(p, index=False),
        overwrite=overwrite,
    )
    def _write_metadata(p: Path) -> None:
        p.write_text(json.dumps(metadata, indent=2))

    _atomic_write_file(metadata_target, _write_metadata, overwrite=overwrite)


def load_signature_scores(path: Path) -> pd.DataFrame | None:
    """Load ``signature_scores.parquet`` from the artifact at ``path``.

    Returns ``None`` if the file does not exist.
    """
    path = Path(path)
    target = path / "signature_scores.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


def load_signature_weights(path: Path) -> SignatureWeights | None:
    """Load ``SignatureWeights`` from ``signature_metadata.json``.

    Returns ``None`` if the file does not exist. Raises
    ``SchemaVersionError`` on metadata version mismatch.
    """
    path = Path(path)
    target = path / "signature_metadata.json"
    if not target.exists():
        return None

    metadata = json.loads(target.read_text())
    actual_version = metadata.get("schema_version")
    if actual_version != SIGNATURE_METADATA_SCHEMA_VERSION:
        raise SchemaVersionError(
            expected=SIGNATURE_METADATA_SCHEMA_VERSION,
            actual=actual_version,
            kind="signature_metadata",
        )

    weights_dict = metadata["weights"]
    return SignatureWeights(**weights_dict)
