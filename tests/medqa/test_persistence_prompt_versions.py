"""Test that signature_metadata.json's optional prompt_versions field
round-trips correctly through save_signature_scores / load_*."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsig.core.graph import AssemblyGraphBuilder, EdgeClass
from bsig.core.persistence import (
    load_signature_weights,
    save_graph,
    save_signature_scores,
)
from bsig.core.signature import SignatureWeights


def _classified_graph():
    b = AssemblyGraphBuilder()
    for n in ("A", "B"):
        b.add_visit(n)
    b.add_transition("A", "x", "B")
    b.set_edge_attributes(
        "A", "x", "B",
        voi=0.5, consensus_rate=0.8, classification=EdgeClass.CONSENSUS,
    )
    return b.build()


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trajectory_id": ["t1"],
            "entropy_plateau": np.array([0.0], dtype=np.float32),
            "voi_flatness": np.array([0.5], dtype=np.float32),
            "distance_from_trajectory": np.array([0.3], dtype=np.float32),
            "composite": np.array([0.4], dtype=np.float32),
        }
    )


def test_prompt_versions_round_trips(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_signature_scores(
        _scores(),
        SignatureWeights(),
        artifact,
        prompt_versions={
            "condition_a": 1,
            "condition_b": 2,
            "condition_c_initial": 1,
            "condition_c_measurement": 1,
            "repair": 0,
        },
    )
    metadata = json.loads(
        (artifact / "signature_metadata.json").read_text()
    )
    assert metadata["prompt_versions"]["condition_a"] == 1
    assert metadata["prompt_versions"]["condition_b"] == 2
    assert metadata["prompt_versions"]["repair"] == 0


def test_prompt_versions_optional_absent_by_default(tmp_path: Path) -> None:
    """Absent prompt_versions is fine; metadata key not present."""
    artifact = tmp_path / "art"
    save_graph(_classified_graph(), artifact)
    save_signature_scores(_scores(), SignatureWeights(), artifact)
    metadata = json.loads(
        (artifact / "signature_metadata.json").read_text()
    )
    assert "prompt_versions" not in metadata
    # And weights still load fine
    weights = load_signature_weights(artifact)
    assert weights is not None
