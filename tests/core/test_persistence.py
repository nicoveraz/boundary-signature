"""Tests for AssemblyGraph persistence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsig.core.graph import (
    AssemblyGraph,
    AssemblyGraphBuilder,
    EdgeClass,
)
from bsig.core.persistence import (
    MANIFEST_SCHEMA_VERSION,
    ArtifactExistsError,
    ArtifactNotFoundError,
    SchemaVersionError,
    load_assembly_indices,
    load_embedding_indices,
    load_graph,
    load_visits,
    save_assembly_indices,
    save_embedding_indices,
    save_graph,
    save_visits,
)


# ---- Fixtures ----


def _classified_graph() -> AssemblyGraph:
    b = AssemblyGraphBuilder(metadata={"experiment": "chest_pain", "n": 3})
    for node in ("A", "B", "C"):
        b.add_visit(node)
    b.add_transition("A", "act1", "B")
    b.add_transition("B", "act2", "C")
    for s, a, t, cls in [
        ("A", "act1", "B", EdgeClass.CONSENSUS),
        ("B", "act2", "C", EdgeClass.RITUALIZED),
    ]:
        b.set_edge_attributes(
            s, a, t, voi=0.42, consensus_rate=0.9, classification=cls,
        )
    return b.build()


def _unclassified_graph() -> AssemblyGraph:
    b = AssemblyGraphBuilder()
    for node in ("X", "Y"):
        b.add_visit(node)
    b.add_transition("X", "step", "Y")
    return b.build()


# ---- save_graph / load_graph round-trip ----


def test_save_load_graph_roundtrip(tmp_path: Path) -> None:
    g = _classified_graph()
    artifact = tmp_path / "art"
    save_graph(g, artifact)
    loaded = load_graph(artifact)

    assert loaded.num_nodes == g.num_nodes
    assert loaded.num_edges == g.num_edges
    assert {n.node_id for n in loaded.iter_nodes()} == {"A", "B", "C"}
    assert loaded.terminal_nodes() == frozenset({"C"})
    assert loaded.metadata["experiment"] == "chest_pain"

    out_a = list(loaded.outgoing_edges("A"))
    assert len(out_a) == 1
    assert out_a[0].target_id == "B"
    assert out_a[0].action_id == "act1"
    assert out_a[0].classification == EdgeClass.CONSENSUS
    assert out_a[0].voi == pytest.approx(0.42, rel=1e-3)


def test_unclassified_graph_roundtrip(tmp_path: Path) -> None:
    g = _unclassified_graph()
    artifact = tmp_path / "art"
    save_graph(g, artifact)
    loaded = load_graph(artifact)
    [edge] = list(loaded.outgoing_edges("X"))
    assert edge.classification == EdgeClass.UNCLASSIFIED


def test_load_graph_via_networkx(tmp_path: Path) -> None:
    """Loaded graph has a working frozen NetworkX view."""
    import networkx as nx
    g = _classified_graph()
    artifact = tmp_path / "art"
    save_graph(g, artifact)
    loaded = load_graph(artifact)
    nx_g = loaded.to_networkx()
    assert nx.is_frozen(nx_g)
    path = nx.shortest_path(nx_g, source="A", target="C")
    assert path == ["A", "B", "C"]


# ---- save_graph: overwrite semantics ----


def test_save_graph_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    g = _classified_graph()
    artifact = tmp_path / "art"
    save_graph(g, artifact)
    with pytest.raises(ArtifactExistsError):
        save_graph(g, artifact)


def test_save_graph_overwrite_true_replaces(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_graph(_unclassified_graph(), artifact, overwrite=True)
    loaded = load_graph(artifact)
    assert loaded.num_nodes == 2
    assert {n.node_id for n in loaded.iter_nodes()} == {"X", "Y"}


def test_save_graph_parent_must_exist(tmp_path: Path) -> None:
    g = _classified_graph()
    nonexistent = tmp_path / "missing" / "art"
    with pytest.raises(ArtifactNotFoundError):
        save_graph(g, nonexistent)


# ---- load_graph error paths ----


def test_load_graph_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        load_graph(tmp_path / "nope")


def test_load_graph_missing_required_file(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    (artifact / "edges.parquet").unlink()
    with pytest.raises(ArtifactNotFoundError, match="edges.parquet"):
        load_graph(artifact)


def test_load_graph_schema_version_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(SchemaVersionError) as excinfo:
        load_graph(artifact)
    assert excinfo.value.actual == 99
    assert excinfo.value.expected == 1
    assert excinfo.value.kind == "graph"


def test_metadata_includes_edge_class_encoding(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    metadata = json.loads((artifact / "metadata.json").read_text())
    encoding = metadata["edge_class_encoding"]
    assert encoding["-1"] == "UNCLASSIFIED"
    assert encoding["0"] == "CONSENSUS"
    assert encoding["1"] == "UNDERUTILIZED"
    assert encoding["2"] == "RITUALIZED"


# ---- visits ----


def _sample_visits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "visit_seq": np.arange(4, dtype=np.int64),
            "trajectory_id": ["t1", "t1", "t2", "t2"],
            "timestep": np.array([0, 1, 0, 1], dtype=np.int32),
            "node_id": ["A", "B", "A", "B"],
            "embedding": [
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([0.3, 0.4], dtype=np.float32),
                np.array([0.5, 0.6], dtype=np.float32),
                np.array([0.7, 0.8], dtype=np.float32),
            ],
        }
    )


def test_save_load_visits_roundtrip(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    visits = _sample_visits()
    save_visits(visits, artifact)
    loaded = load_visits(artifact)
    assert loaded is not None
    assert list(loaded.columns) == [
        "visit_seq", "trajectory_id", "timestep", "node_id", "embedding"
    ]
    assert loaded["visit_seq"].tolist() == [0, 1, 2, 3]


def test_load_visits_returns_none_when_absent(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    assert load_visits(artifact) is None


def test_save_visits_validates_required_columns(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    bad = _sample_visits().drop(columns=["embedding"])
    with pytest.raises(ValueError, match="missing required columns"):
        save_visits(bad, artifact)


def test_save_visits_rejects_duplicate_visit_seq(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    visits = _sample_visits()
    visits.loc[3, "visit_seq"] = 0
    with pytest.raises(ValueError, match="visit_seq must be unique"):
        save_visits(visits, artifact)


def test_save_visits_overwrite_semantics(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_visits(_sample_visits(), artifact)
    with pytest.raises(ArtifactExistsError):
        save_visits(_sample_visits(), artifact)
    save_visits(_sample_visits(), artifact, overwrite=True)


def test_save_visits_requires_artifact_dir(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        save_visits(_sample_visits(), tmp_path / "nope")


# ---- assembly indices ----


def _sample_assembly_indices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "assembly_index": np.array([0, 1, 2], dtype=np.int32),
        }
    )


def test_save_load_assembly_indices_roundtrip(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_assembly_indices(_sample_assembly_indices(), artifact)
    loaded = load_assembly_indices(artifact)
    assert loaded is not None
    assert list(loaded.columns) == ["node_id", "assembly_index"]
    assert loaded["assembly_index"].tolist() == [0, 1, 2]


def test_load_assembly_indices_returns_none_when_absent(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    assert load_assembly_indices(artifact) is None


def test_save_assembly_indices_validates_columns(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    bad = _sample_assembly_indices().drop(columns=["assembly_index"])
    with pytest.raises(ValueError, match="missing required columns"):
        save_assembly_indices(bad, artifact)


def test_save_assembly_indices_overwrite_semantics(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_assembly_indices(_sample_assembly_indices(), artifact)
    with pytest.raises(ArtifactExistsError):
        save_assembly_indices(_sample_assembly_indices(), artifact)
    save_assembly_indices(
        _sample_assembly_indices(), artifact, overwrite=True
    )


# ---- atomic-write semantics ----


def test_save_graph_does_not_leave_tmp_dirs(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    siblings = list(tmp_path.iterdir())
    assert len(siblings) == 1
    assert siblings[0].name == "art"


def test_save_graph_failed_overwrite_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename fails mid-overwrite, the original artifact survives."""
    import os

    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    original_files = sorted(p.name for p in artifact.iterdir())

    real_rename = os.rename
    rename_calls = {"n": 0}

    def flaky_rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        rename_calls["n"] += 1
        # The first rename moves the existing target out of the way (succeeds);
        # the second rename moves tmp into place — fail this one to trigger
        # the restore path.
        if rename_calls["n"] == 2:
            raise OSError("simulated rename failure")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", flaky_rename)

    with pytest.raises(OSError, match="simulated rename failure"):
        save_graph(_unclassified_graph(), artifact, overwrite=True)

    # Original artifact must still be loadable
    monkeypatch.setattr(os, "rename", real_rename)
    loaded = load_graph(artifact)
    assert {n.node_id for n in loaded.iter_nodes()} == {"A", "B", "C"}
    # No leftover tmp/old siblings
    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["art"]
    # Files inside untouched
    assert sorted(p.name for p in artifact.iterdir()) == original_files


# ---- FAISS embedding indices (skip if faiss not installed) ----


def test_save_load_embedding_indices_roundtrip(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)

    dim = 4
    indices = {}
    for ts in (0, 1, 2):
        idx = faiss.IndexFlatIP(dim)
        vectors = np.random.RandomState(ts).randn(5, dim).astype(np.float32)
        idx.add(vectors)
        indices[ts] = idx

    manifest = {
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_revision": "abc123",
        "dimension": dim,
        "normalization": "l2",
        "index_type": "IndexFlatIP",
    }
    save_embedding_indices(indices, artifact, manifest)
    loaded = load_embedding_indices(artifact)
    assert set(loaded) == {0, 1, 2}
    for ts, idx in loaded.items():
        assert idx.ntotal == 5

    # Manifest schema version was injected
    manifest_path = artifact / "faiss_indices" / "manifest.json"
    saved = json.loads(manifest_path.read_text())
    assert saved["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert saved["timesteps"] == {"0": 5, "1": 5, "2": 5}
    assert saved["index_type"] == "IndexFlatIP"


def test_load_embedding_indices_returns_empty_when_absent(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    assert load_embedding_indices(artifact) == {}


def test_load_embedding_indices_schema_version_mismatch(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)

    idx = faiss.IndexFlatIP(2)
    idx.add(np.zeros((1, 2), dtype=np.float32))
    save_embedding_indices(
        {0: idx}, artifact,
        {
            "embedding_model": "x", "dimension": 2,
            "normalization": "l2", "index_type": "IndexFlatIP",
        },
    )

    manifest_path = artifact / "faiss_indices" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SchemaVersionError) as excinfo:
        load_embedding_indices(artifact)
    assert excinfo.value.kind == "faiss_manifest"


def test_save_embedding_indices_requires_artifact_dir(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    with pytest.raises(ArtifactNotFoundError):
        save_embedding_indices({}, tmp_path / "nope", {"index_type": "x"})


# ---- signature scores ----


def _sample_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trajectory_id": ["t1", "t2", "t3"],
            "entropy_plateau": np.array([-0.1, 0.0, 0.5], dtype=np.float32),
            "voi_flatness": np.array([0.3, 0.4, 0.5], dtype=np.float32),
            "distance_from_trajectory": np.array(
                [0.1, 0.5, 0.9], dtype=np.float32
            ),
            "composite": np.array([0.2, 0.5, 0.8], dtype=np.float32),
        }
    )


def test_save_load_signature_scores_roundtrip(tmp_path: Path) -> None:
    from bsig.core.signature import SignatureWeights
    from bsig.core.persistence import (
        load_signature_scores,
        load_signature_weights,
        save_signature_scores,
    )
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    scores = _sample_scores()
    weights = SignatureWeights(0.5, 0.25, 0.25)
    save_signature_scores(scores, weights, artifact)

    loaded_scores = load_signature_scores(artifact)
    loaded_weights = load_signature_weights(artifact)
    assert loaded_scores is not None
    assert loaded_weights is not None
    assert loaded_scores["trajectory_id"].tolist() == ["t1", "t2", "t3"]
    assert loaded_weights.entropy_plateau == pytest.approx(0.5)


def test_load_signature_scores_returns_none_when_absent(tmp_path: Path) -> None:
    from bsig.core.persistence import load_signature_scores
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    assert load_signature_scores(artifact) is None


def test_load_signature_weights_returns_none_when_absent(tmp_path: Path) -> None:
    from bsig.core.persistence import load_signature_weights
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    assert load_signature_weights(artifact) is None


def test_save_signature_scores_validates_columns(tmp_path: Path) -> None:
    from bsig.core.persistence import save_signature_scores
    from bsig.core.signature import SignatureWeights
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    bad = _sample_scores().drop(columns=["composite"])
    with pytest.raises(ValueError, match="missing required columns"):
        save_signature_scores(bad, SignatureWeights(), artifact)


def test_save_signature_scores_overwrite_semantics(tmp_path: Path) -> None:
    from bsig.core.persistence import save_signature_scores
    from bsig.core.signature import SignatureWeights
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_signature_scores(_sample_scores(), SignatureWeights(), artifact)
    with pytest.raises(ArtifactExistsError):
        save_signature_scores(_sample_scores(), SignatureWeights(), artifact)
    save_signature_scores(
        _sample_scores(), SignatureWeights(), artifact, overwrite=True
    )


def test_load_signature_weights_schema_version_mismatch(tmp_path: Path) -> None:
    from bsig.core.persistence import (
        load_signature_weights,
        save_signature_scores,
    )
    from bsig.core.signature import SignatureWeights
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_signature_scores(_sample_scores(), SignatureWeights(), artifact)

    metadata_path = artifact / "signature_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(SchemaVersionError) as excinfo:
        load_signature_weights(artifact)
    assert excinfo.value.kind == "signature_metadata"


def test_signature_metadata_records_normalization_and_path(tmp_path: Path) -> None:
    from bsig.core.persistence import save_signature_scores
    from bsig.core.signature import SignatureWeights
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    graph_ref = tmp_path / "graph_ref"
    save_signature_scores(
        _sample_scores(),
        SignatureWeights(),
        artifact,
        graph_artifact_path=graph_ref,
    )
    metadata = json.loads(
        (artifact / "signature_metadata.json").read_text()
    )
    assert metadata["normalization"] == "rank_percentile"
    assert metadata["graph_artifact_path"] == str(graph_ref)
    assert metadata["n_trajectories"] == 3
