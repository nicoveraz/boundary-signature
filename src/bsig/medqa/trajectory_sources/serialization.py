"""Cached-trajectories serialization for the pre-recovered source.

The cached-trajectories format stores Trajectory objects in a two-table
Parquet layout. It is the persistence backing for
``MedQAPrerecoveredTrajectorySource`` (reader) and Condition C's
output writer (stage 3.3 work).

Format scope: trajectory data only (states + actions + outcome).
Recovered-graph data (VoI, consensus_rate, classification, FAISS
indices) lives in the stage-2.2 graph artifact format, which is a
DIFFERENT artifact concept. The two formats are not interchangeable;
do not attempt to merge them.

Constraint (per design pass Q6): values in
``Outcome.secondary_labels``, ``State.metadata``, and
``Action.metadata`` must be JSON-serializable when written through
this format. The writer raises ``ValueError`` with a clear message on
violation. Domain-pack ground-truth extractors that produce non-JSON-
serializable secondary labels (e.g., numpy arrays, Pydantic models,
dataclass instances) must serialize them to JSON-compatible primitives
before constructing the Outcome.

Layout:

    artifact_dir/
    ├── metadata.json           schema_version, created_at, optional source/condition
    ├── trajectories.parquet    one row per trajectory
    │                           (trajectory_id PK, primary_label,
    │                            confidence, secondary_labels_json,
    │                            n_states, n_actions)
    └── states.parquet          one row per state, with action data
                                inline on the from-state
                                ((trajectory_id, position) PK; node_id,
                                 timestep, embedding, metadata_json,
                                 hypothesis_distribution_json,
                                 mass_capture (v2+),
                                 top_k_logprobs_json (v3+),
                                 action_id_to_next, action_category_
                                 to_next, action_metadata_to_next_json)

schema_version = 3 (current); writer always emits v3.

Schema history:
- v1: pre-ADR-0008. ``hypothesis_distribution`` came from verbalised
  distributions; no per-state mass_capture concept.
- v2: post-ADR-0008. ``mass_capture`` column added; reflects the
  structured field on ``State`` for token-probability measurements.
- v3: post-stage-4a-replication. ``top_k_logprobs_json`` column
  added; preserves the raw top-K next-token logprobs at each
  measurement position (the *measurement* in measurement-vs-computation
  terminology). Storage at full fidelity so downstream computations
  (alternative entropy summaries, sensitivity analyses, prompt-variant
  retrospectives) can be re-derived without re-running model inference.
  Reader accepts v1, v2, v3; v1/v2 trajectories load with
  ``State.top_k_logprobs = None``.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bsig.core.persistence import (
    ArtifactExistsError,
    ArtifactNotFoundError,
    SchemaVersionError,
)
from bsig.core.trajectory import Action, Outcome, State, Trajectory


CACHED_TRAJECTORIES_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)


def _to_json(value: Any, *, where: str) -> str:
    """Serialize ``value`` to JSON or raise a clear error."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Object at {where} is not JSON-serializable "
            f"(type={type(value).__name__}). The cached-trajectories "
            f"format requires JSON-serializable values in "
            f"Outcome.secondary_labels, State.metadata, and "
            f"Action.metadata. Convert to dicts/lists/strings/numbers "
            f"before constructing the Outcome / State / Action."
        ) from exc


def save_cached_trajectories(
    trajectories: Sequence[Trajectory],
    path: Path,
    *,
    overwrite: bool = False,
    source_dataset: str | None = None,
    condition_id: str | None = None,
) -> None:
    """Write a sequence of Trajectory objects to ``path/`` as
    ``trajectories.parquet`` + ``states.parquet`` + ``metadata.json``.

    Atomic write semantics: directory is created in a tmp location and
    renamed on success (parallel to ``save_graph`` from stage 2.2).

    Raises:
    - ``ArtifactExistsError`` if ``path`` exists and ``overwrite`` is
      False.
    - ``ValueError`` if any Outcome/State/Action carries a non-JSON-
      serializable metadata or secondary_labels value (per Q6).
    """
    from bsig.core.persistence import _atomic_write_dir  # noqa: PLC0415

    path = Path(path)

    traj_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []

    for traj in trajectories:
        outcome = traj.outcome
        if outcome is not None:
            secondary_json = _to_json(
                dict(outcome.secondary_labels),
                where=f"trajectory {traj.trajectory_id!r} outcome.secondary_labels",
            )
            primary_label: str | None = outcome.primary_label
            confidence: float | None = outcome.confidence
        else:
            secondary_json = None
            primary_label = None
            confidence = None

        traj_rows.append(
            {
                "trajectory_id": traj.trajectory_id,
                "primary_label": primary_label,
                "confidence": confidence,
                "secondary_labels_json": secondary_json,
                "n_states": len(traj.states),
                "n_actions": len(traj.actions),
            }
        )

        for position, state in enumerate(traj.states):
            metadata_json = _to_json(
                dict(state.metadata),
                where=(
                    f"trajectory {traj.trajectory_id!r} state[{position}] "
                    f".metadata"
                ),
            )
            hd_json: str | None
            if state.hypothesis_distribution is not None:
                hd_json = _to_json(
                    dict(state.hypothesis_distribution),
                    where=(
                        f"trajectory {traj.trajectory_id!r} state[{position}] "
                        f".hypothesis_distribution"
                    ),
                )
            else:
                hd_json = None
            embedding_list: list[float] | None
            if state.embedding is not None:
                embedding_list = state.embedding.astype(np.float32).tolist()
            else:
                embedding_list = None

            if position < len(traj.actions):
                action = traj.actions[position]
                action_metadata_json = _to_json(
                    dict(action.metadata),
                    where=(
                        f"trajectory {traj.trajectory_id!r} action[{position}] "
                        f".metadata"
                    ),
                )
                action_id_to_next: str | None = action.action_id
                action_category_to_next: str | None = action.action_category
                action_metadata_to_next_json: str | None = action_metadata_json
            else:
                action_id_to_next = None
                action_category_to_next = None
                action_metadata_to_next_json = None

            top_k_json: str | None
            if state.top_k_logprobs is not None:
                top_k_json = _to_json(
                    dict(state.top_k_logprobs),
                    where=(
                        f"trajectory {traj.trajectory_id!r} state[{position}] "
                        f".top_k_logprobs"
                    ),
                )
            else:
                top_k_json = None
            state_rows.append(
                {
                    "trajectory_id": traj.trajectory_id,
                    "position": position,
                    "node_id": state.node_id,
                    "timestep": state.timestep,
                    "embedding": embedding_list,
                    "metadata_json": metadata_json,
                    "hypothesis_distribution_json": hd_json,
                    "mass_capture": (
                        float(state.mass_capture)
                        if state.mass_capture is not None
                        else None
                    ),
                    "top_k_logprobs_json": top_k_json,
                    "action_id_to_next": action_id_to_next,
                    "action_category_to_next": action_category_to_next,
                    "action_metadata_to_next_json": action_metadata_to_next_json,
                }
            )

    metadata = {
        "schema_version": CACHED_TRAJECTORIES_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_trajectories": len(trajectories),
        "source_dataset": source_dataset,
        "condition_id": condition_id,
    }

    traj_df = _build_traj_df(traj_rows)
    states_df = _build_states_df(state_rows)

    with _atomic_write_dir(path, overwrite=overwrite) as tmp:
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2))
        traj_df.to_parquet(tmp / "trajectories.parquet", index=False)
        states_df.to_parquet(tmp / "states.parquet", index=False)


def load_cached_trajectories(path: Path) -> list[Trajectory]:
    """Read trajectories from a cached-trajectories artifact directory.

    Raises:
    - ``ArtifactNotFoundError`` if any required file is missing.
    - ``SchemaVersionError`` if metadata.schema_version != current.
    """
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise ArtifactNotFoundError(
            f"Cached-trajectories artifact directory not found: {path}"
        )

    metadata_path = path / "metadata.json"
    traj_path = path / "trajectories.parquet"
    states_path = path / "states.parquet"
    for required in (metadata_path, traj_path, states_path):
        if not required.exists():
            raise ArtifactNotFoundError(f"Required file missing: {required}")

    metadata = json.loads(metadata_path.read_text())
    actual_version = metadata.get("schema_version")
    if actual_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            expected=CACHED_TRAJECTORIES_SCHEMA_VERSION,
            actual=actual_version,
            kind="cached_trajectories",
        )

    traj_df = pd.read_parquet(traj_path)
    states_df = pd.read_parquet(states_path).sort_values(
        ["trajectory_id", "position"]
    )

    states_by_traj: dict[str, list[dict[str, Any]]] = {}
    for row in states_df.to_dict(orient="records"):
        states_by_traj.setdefault(row["trajectory_id"], []).append(row)

    trajectories: list[Trajectory] = []
    for traj_row in traj_df.to_dict(orient="records"):
        tid = traj_row["trajectory_id"]
        state_dicts = states_by_traj.get(tid, [])
        states = tuple(_build_state(d) for d in state_dicts)
        actions = tuple(
            _build_action(d)
            for d in state_dicts
            if not _is_null(d["action_id_to_next"])
        )
        outcome = _build_outcome(traj_row)
        trajectories.append(
            Trajectory(
                trajectory_id=tid,
                states=states,
                actions=actions,
                outcome=outcome,
            )
        )

    return trajectories


def iter_cached_trajectories(path: Path) -> Iterator[Trajectory]:
    """Streaming reader. Currently materializes all trajectories then
    yields — at 1273-trajectory MedQA scale this is fine. If the
    framework ever caches at clinical scale (~25k trajectories), revise
    to stream from the Parquet files directly.
    """
    yield from load_cached_trajectories(path)


def _build_traj_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                "trajectory_id": pd.Series(dtype=object),
                "primary_label": pd.Series(dtype=object),
                "confidence": pd.Series(dtype=np.float32),
                "secondary_labels_json": pd.Series(dtype=object),
                "n_states": pd.Series(dtype=np.int32),
                "n_actions": pd.Series(dtype=np.int32),
            }
        )
    df = pd.DataFrame(rows)
    df["confidence"] = df["confidence"].astype("Float32")
    df["n_states"] = df["n_states"].astype(np.int32)
    df["n_actions"] = df["n_actions"].astype(np.int32)
    return df


def _build_states_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                "trajectory_id": pd.Series(dtype=object),
                "position": pd.Series(dtype=np.int32),
                "node_id": pd.Series(dtype=object),
                "timestep": pd.Series(dtype=np.int32),
                "embedding": pd.Series(dtype=object),
                "metadata_json": pd.Series(dtype=object),
                "hypothesis_distribution_json": pd.Series(dtype=object),
                "mass_capture": pd.Series(dtype="Float32"),
                "top_k_logprobs_json": pd.Series(dtype=object),
                "action_id_to_next": pd.Series(dtype=object),
                "action_category_to_next": pd.Series(dtype=object),
                "action_metadata_to_next_json": pd.Series(dtype=object),
            }
        )
    df = pd.DataFrame(rows)
    df["position"] = df["position"].astype(np.int32)
    df["timestep"] = df["timestep"].astype(np.int32)
    df["mass_capture"] = df["mass_capture"].astype("Float32")
    return df


def _is_null(value: Any) -> bool:
    """Pandas Parquet roundtrip turns None into either None or NaN
    depending on column dtype; check both."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _build_state(row: Mapping[str, Any]) -> State:
    embedding_value = row["embedding"]
    embedding: np.ndarray | None
    if _is_null(embedding_value):
        embedding = None
    else:
        embedding = np.asarray(embedding_value, dtype=np.float32)
    metadata_json = row["metadata_json"]
    metadata = (
        json.loads(metadata_json) if not _is_null(metadata_json) else {}
    )
    hd_json = row["hypothesis_distribution_json"]
    hd = json.loads(hd_json) if not _is_null(hd_json) else None
    # mass_capture is v2+; absent (key missing) for v1-cached
    # trajectories, or null per-row for v2 trajectories produced by
    # measurement protocols that don't compute it.
    mass_capture: float | None
    mc_value = row["mass_capture"] if "mass_capture" in row else None
    if mc_value is None or _is_null(mc_value):
        mass_capture = None
    else:
        mass_capture = float(mc_value)
    # top_k_logprobs is v3+; absent (key missing) for v1/v2-cached
    # trajectories, or null per-row for v3 trajectories produced by
    # adapters that don't expose top-K logprobs.
    top_k: Mapping[str, float] | None
    tk_value = row["top_k_logprobs_json"] if "top_k_logprobs_json" in row else None
    if tk_value is None or _is_null(tk_value):
        top_k = None
    else:
        top_k = json.loads(tk_value)
    return State(
        node_id=str(row["node_id"]),
        timestep=int(row["timestep"]),
        embedding=embedding,
        metadata=metadata,
        hypothesis_distribution=hd,
        mass_capture=mass_capture,
        top_k_logprobs=top_k,
    )


def _build_action(row: Mapping[str, Any]) -> Action:
    metadata_json = row["action_metadata_to_next_json"]
    metadata = (
        json.loads(metadata_json) if not _is_null(metadata_json) else {}
    )
    category = row["action_category_to_next"]
    return Action(
        action_id=str(row["action_id_to_next"]),
        action_category=None if _is_null(category) else str(category),
        metadata=metadata,
    )


def _build_outcome(row: Mapping[str, Any]) -> Outcome | None:
    if _is_null(row["primary_label"]):
        return None
    secondary_json = row["secondary_labels_json"]
    secondary = (
        json.loads(secondary_json) if not _is_null(secondary_json) else {}
    )
    return Outcome(
        primary_label=str(row["primary_label"]),
        confidence=float(row["confidence"]),
        secondary_labels=secondary,
    )
