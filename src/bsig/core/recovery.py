"""Trace-based recovery: trajectories -> AssemblyGraph + visits.

This module is the assembly-space side of the framework's pipeline.
``bsig.core`` is the assembly-space module of the synthesis architecture
(equivalence, granularity, distance + assembly graph + per-leaf metrics).
The concept-space integration enters via ``StateCanonicalizer``
implementations using ``EmbeddingSource`` under the hood — i.e., a
canonicalizer that uses the embedding source to compute discrete node IDs
satisfies the ``StateCanonicalizer`` contract without any change here.

What recovery is NOT responsible for:

- **Edge granularity** (whether each observation is its own action or
  whether observations are batched into composite actions). Granularity
  is fixed at trajectory-construction time by the upstream
  ``TrajectorySource``. Recovery sees actions as constructed.
- **Time as a graph dimension** (whether wall-clock intervals between
  states matter). Time is a state-representation concern: include it
  in ``State.metadata`` and have the canonicalizer use it, or do not.
  Recovery treats states as opaque ``node_id`` values.
- **Subpopulation analysis**. The recovered graph aggregates over the
  full input. Slicing by subpopulation is a downstream concern — run
  recovery on the slice.

What recovery IS responsible for:

1. Validating input consistency (state metadata is canonical for each
   ``node_id``; embeddings are all-present or all-absent).
2. Aggregating visits and transitions across trajectories.
3. Computing action-level VoI and consensus_rate (both are properties
   of ``(source_id, action_id)``; replicated to all edges sharing that
   pair for I/O convenience).
4. Classifying edges via percentile thresholds on VoI and consensus_rate.
5. Building the immutable ``AssemblyGraph`` and its companion
   ``visits`` DataFrame.

VoI prior fallback (three-tier for posterior, two-tier for prior):

- **Prior** ``H(D | s)``: local (outcomes among trajectories that
  visited s) if visit count ≥ ``voi_local_prior_min_count``, else
  global (outcomes across all labeled trajectories).
- **Posterior** ``H(D | t', via (s, a))``: local-via-edge (outcomes
  among trajectories that took (s, a, t')) if count ≥ threshold,
  else local (outcomes among trajectories that visited t'), else
  global.

Per-edge fallback levels are not stored on edges in 0.1; coarse-grained
counts are stored in ``graph.metadata["voi_method_summary"]`` so
sensitivity analyses can detect when fallbacks dominated. Per-edge
tracking is deferred to a future schema_version.

Determinism: given the same input trajectory order, recovery produces
bit-identical output. Bit-identity across different input orders of the
same trajectory set is NOT guaranteed (would require explicit sort
overhead callers may not need). Callers wanting cross-run reproducibility
should sort their ``TrajectorySource`` output before passing here.

Unlabeled trajectories (``Trajectory.outcome is None``) contribute to
visit and transition counts but do not contribute to VoI computations.
A graph recovered from purely unlabeled trajectories has no VoI signal
and all edges fall to the global-prior fallback (which is itself
undefined when the global label set is empty — recovery raises in that
case).

Self-loops (``source_id == target_id``) are allowed. Their VoI is 0 by
construction (source and target diagnostic distributions are identical
when they refer to the same canonical node). Classification typically
puts them in RITUALIZED or RARE depending on consensus_rate. This is
expected, not a bug.
"""
from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bsig.core.graph import (
    AssemblyGraph,
    AssemblyGraphBuilder,
    EdgeClass,
)
from bsig.core.trajectory import State, Trajectory


# ---- Public types ----


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Recovery aggregation and classification parameters.

    All percentiles are in [0, 100]. Defaults: top-25% / bottom-25%
    carve-outs with a middle band that gets ``EdgeClass.MIDDLE``.

    The ``classification_*_high_percentile`` and
    ``classification_*_low_percentile`` together with the corresponding
    ``MIDDLE`` band define a 3x3 cell partition on (VoI, consensus_rate).
    Edges in the four corners get the four named classes; every other
    cell becomes MIDDLE.

    Validation:
    - ``0 <= voi_low_percentile < voi_high_percentile <= 100``
    - ``0 <= consensus_low_percentile < consensus_high_percentile <= 100``
    - ``voi_local_prior_min_count >= 1``
    - ``drop_edges_below_frequency >= 1`` (1 means keep all edges)
    """

    voi_local_prior_min_count: int = 30
    classification_voi_high_percentile: float = 75.0
    classification_voi_low_percentile: float = 25.0
    classification_consensus_high_percentile: float = 75.0
    classification_consensus_low_percentile: float = 25.0
    drop_edges_below_frequency: int = 1
    include_self_loops: bool = True

    def __post_init__(self) -> None:
        if self.voi_local_prior_min_count < 1:
            raise ValueError(
                f"voi_local_prior_min_count must be >= 1, "
                f"got {self.voi_local_prior_min_count}"
            )
        if self.drop_edges_below_frequency < 1:
            raise ValueError(
                f"drop_edges_below_frequency must be >= 1, "
                f"got {self.drop_edges_below_frequency}"
            )
        if not (
            0.0
            <= self.classification_voi_low_percentile
            < self.classification_voi_high_percentile
            <= 100.0
        ):
            raise ValueError(
                "voi percentiles must satisfy "
                "0 <= low < high <= 100; got "
                f"low={self.classification_voi_low_percentile}, "
                f"high={self.classification_voi_high_percentile}"
            )
        if not (
            0.0
            <= self.classification_consensus_low_percentile
            < self.classification_consensus_high_percentile
            <= 100.0
        ):
            raise ValueError(
                "consensus percentiles must satisfy "
                "0 <= low < high <= 100; got "
                f"low={self.classification_consensus_low_percentile}, "
                f"high={self.classification_consensus_high_percentile}"
            )


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Output of ``recover_assembly_graph``.

    ``visits`` follows the schema documented in ``persistence.save_visits``:
    columns ``visit_seq`` (int64 PK, sequential from 0), ``trajectory_id``,
    ``timestep`` (int32), ``node_id``, ``embedding`` (list<float32>).
    Empty (zero rows) when input trajectories had no embeddings.
    """

    graph: AssemblyGraph
    visits: pd.DataFrame


# ---- Public entry point ----


def recover_assembly_graph(
    trajectories: Sequence[Trajectory],
    config: RecoveryConfig,
) -> RecoveryResult:
    """Recover an AssemblyGraph from observed trajectories.

    Three internal passes:

    1. Validate input consistency (metadata canonical per node; embeddings
       all-or-none).
    2. Aggregate visits, transitions, and per-state / per-edge outcome
       lists.
    3. Compute action-level VoI and consensus_rate, classify edges by
       percentile, build the graph, assemble the visits DataFrame.

    Raises ``ValueError`` on inconsistent metadata, mixed embedding
    presence, empty input with non-empty config requirements, or other
    structural validation failures.
    """
    has_embeddings = _validate(trajectories)
    aggregated = _aggregate(trajectories, has_embeddings)
    return _build(aggregated, config, has_embeddings)


# ---- Internal: validation pass ----


def _validate(trajectories: Sequence[Trajectory]) -> bool:
    """Eager validation. Returns True if all states have embeddings,
    False if all states have no embeddings. Raises on mixed."""
    seen_metadata: dict[str, Mapping[str, Any]] = {}
    embeddings_state: bool | None = None

    for traj in trajectories:
        for state in traj.states:
            has_emb = state.embedding is not None
            if embeddings_state is None:
                embeddings_state = has_emb
            elif embeddings_state != has_emb:
                raise ValueError(
                    f"Trajectory {traj.trajectory_id!r} timestep "
                    f"{state.timestep} has embedding="
                    f"{'present' if has_emb else 'absent'} but earlier "
                    f"states had embedding="
                    f"{'present' if embeddings_state else 'absent'}. "
                    f"Recovery requires embeddings to be uniformly "
                    f"present or uniformly absent."
                )

            prev = seen_metadata.get(state.node_id)
            if prev is None:
                seen_metadata[state.node_id] = state.metadata
            elif not _metadata_equal(prev, state.metadata):
                diff_keys = sorted(_metadata_diff_keys(prev, state.metadata))
                raise ValueError(
                    f"State {state.node_id!r} has inconsistent metadata "
                    f"across trajectories. Differing keys: {diff_keys}. "
                    f"This indicates a canonicalizer or upstream-data "
                    f"bug — equivalent canonical states must have "
                    f"equivalent metadata."
                )

    return embeddings_state if embeddings_state is not None else False


def _metadata_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if set(a) != set(b):
        return False
    return all(a[k] == b[k] for k in a)


def _metadata_diff_keys(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> set[str]:
    keys_a, keys_b = set(a), set(b)
    return (keys_a ^ keys_b) | {k for k in keys_a & keys_b if a[k] != b[k]}


# ---- Internal: aggregation pass ----


@dataclass(slots=True)
class _Aggregated:
    node_visits: dict[str, int]
    transitions: dict[tuple[str, str, str], int]   # (source, action, target) -> freq
    node_to_outcomes: dict[str, list[str]]
    transition_to_outcomes: dict[tuple[str, str, str], list[str]]
    visits_rows: list[dict[str, Any]]
    global_outcomes: list[str]


def _aggregate(
    trajectories: Sequence[Trajectory], has_embeddings: bool
) -> _Aggregated:
    node_visits: dict[str, int] = defaultdict(int)
    transitions: dict[tuple[str, str, str], int] = defaultdict(int)
    node_to_outcomes: dict[str, list[str]] = defaultdict(list)
    transition_to_outcomes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    visits_rows: list[dict[str, Any]] = []
    global_outcomes: list[str] = []
    visit_seq = 0

    for traj in trajectories:
        outcome_label: str | None = (
            traj.outcome.primary_label if traj.outcome is not None else None
        )
        if outcome_label is not None:
            global_outcomes.append(outcome_label)

        for state in traj.states:
            node_visits[state.node_id] += 1
            if outcome_label is not None:
                node_to_outcomes[state.node_id].append(outcome_label)
            if has_embeddings:
                visits_rows.append(
                    {
                        "visit_seq": visit_seq,
                        "trajectory_id": traj.trajectory_id,
                        "timestep": state.timestep,
                        "node_id": state.node_id,
                        "embedding": state.embedding,
                    }
                )
                visit_seq += 1

        for i, action in enumerate(traj.actions):
            s = traj.states[i].node_id
            t = traj.states[i + 1].node_id
            a = action.action_id
            key = (s, a, t)
            transitions[key] += 1
            if outcome_label is not None:
                transition_to_outcomes[key].append(outcome_label)

    return _Aggregated(
        node_visits=dict(node_visits),
        transitions=dict(transitions),
        node_to_outcomes=dict(node_to_outcomes),
        transition_to_outcomes=dict(transition_to_outcomes),
        visits_rows=visits_rows,
        global_outcomes=global_outcomes,
    )


# ---- Internal: action-level metrics + classification + build ----


def _shannon_entropy(labels: list[str]) -> float:
    """Shannon entropy in bits. Returns 0.0 for empty or single-label sets."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _build(
    agg: _Aggregated, config: RecoveryConfig, has_embeddings: bool
) -> RecoveryResult:
    # Filter transitions by frequency and self-loop policy
    filtered: dict[tuple[str, str, str], int] = {}
    for (s, a, t), freq in agg.transitions.items():
        if freq < config.drop_edges_below_frequency:
            continue
        if not config.include_self_loops and s == t:
            continue
        filtered[(s, a, t)] = freq

    # Compute prior entropy per state with two-tier fallback
    prior_h: dict[str, float] = {}
    prior_level_counts = {"local": 0, "global": 0}
    h_global = _shannon_entropy(agg.global_outcomes)
    for node_id in agg.node_visits:
        outcomes = agg.node_to_outcomes.get(node_id, [])
        if len(outcomes) >= config.voi_local_prior_min_count:
            prior_h[node_id] = _shannon_entropy(outcomes)
            prior_level_counts["local"] += 1
        else:
            prior_h[node_id] = h_global
            prior_level_counts["global"] += 1

    # Group filtered edges by (source, action) for action-level metrics
    edges_by_sa: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (s, a, t), freq in filtered.items():
        edges_by_sa[(s, a)].append((t, freq))

    # Compute action-level VoI and consensus_rate
    action_voi: dict[tuple[str, str], float] = {}
    action_consensus: dict[tuple[str, str], float] = {}
    posterior_level_counts = {"local_via_edge": 0, "local": 0, "global": 0}

    for (s, a), targets in edges_by_sa.items():
        total_freq_sa = sum(f for _, f in targets)
        # consensus_rate = P(action a is taken | at state s)
        action_consensus[(s, a)] = total_freq_sa / agg.node_visits[s]

        # Posterior weighted sum with three-tier fallback per target
        h_post_avg = 0.0
        for t, freq in targets:
            p_t = freq / total_freq_sa
            via_outcomes = agg.transition_to_outcomes.get((s, a, t), [])
            if len(via_outcomes) >= config.voi_local_prior_min_count:
                h_t = _shannon_entropy(via_outcomes)
                posterior_level_counts["local_via_edge"] += 1
            else:
                target_outcomes = agg.node_to_outcomes.get(t, [])
                if len(target_outcomes) >= config.voi_local_prior_min_count:
                    h_t = _shannon_entropy(target_outcomes)
                    posterior_level_counts["local"] += 1
                else:
                    h_t = h_global
                    posterior_level_counts["global"] += 1
            h_post_avg += p_t * h_t

        # VoI = H(D | s) - E_t [ H(D | t, via s,a) ]
        action_voi[(s, a)] = prior_h[s] - h_post_avg

    # Classify by percentile cells (only meaningful with at least one (s, a))
    classification: dict[tuple[str, str], EdgeClass] = {}
    if action_voi:
        voi_array = np.array(list(action_voi.values()), dtype=np.float64)
        cons_array = np.array(list(action_consensus.values()), dtype=np.float64)
        voi_high = float(
            np.percentile(voi_array, config.classification_voi_high_percentile)
        )
        voi_low = float(
            np.percentile(voi_array, config.classification_voi_low_percentile)
        )
        cons_high = float(
            np.percentile(
                cons_array, config.classification_consensus_high_percentile
            )
        )
        cons_low = float(
            np.percentile(
                cons_array, config.classification_consensus_low_percentile
            )
        )
        for sa, voi in action_voi.items():
            consensus = action_consensus[sa]
            classification[sa] = _classify(
                voi, consensus, voi_high, voi_low, cons_high, cons_low
            )

    # Build the graph
    builder_metadata: dict[str, Any] = {
        "recovery_config": dataclasses.asdict(config),
        "voi_method_summary": {
            "prior": prior_level_counts,
            "posterior": posterior_level_counts,
        },
        "voi_global_entropy": h_global,
    }
    builder = AssemblyGraphBuilder(metadata=builder_metadata)

    for node_id, count in agg.node_visits.items():
        builder.add_visit(node_id, count=count)

    for (s, a, t), freq in filtered.items():
        builder.add_transition(s, a, t, count=freq)

    for (s, a, t) in filtered:
        voi = action_voi[(s, a)]
        consensus = action_consensus[(s, a)]
        cls = classification.get((s, a), EdgeClass.UNCLASSIFIED)
        builder.set_edge_attributes(
            s, a, t, voi=voi, consensus_rate=consensus, classification=cls
        )

    graph = builder.build()
    visits_df = _build_visits_df(agg.visits_rows, has_embeddings)
    return RecoveryResult(graph=graph, visits=visits_df)


def _classify(
    voi: float,
    consensus: float,
    voi_high: float,
    voi_low: float,
    cons_high: float,
    cons_low: float,
) -> EdgeClass:
    """Map (voi, consensus) to a 3x3 cell of the percentile partition."""
    voi_is_high = voi >= voi_high
    voi_is_low = voi <= voi_low
    cons_is_high = consensus >= cons_high
    cons_is_low = consensus <= cons_low

    # Disambiguate degenerate case where high == low (single-edge graph etc.):
    # treat as high to keep the corner classes populated.
    if voi_is_high and cons_is_high:
        return EdgeClass.CONSENSUS
    if voi_is_high and cons_is_low:
        return EdgeClass.UNDERUTILIZED
    if voi_is_low and cons_is_high:
        return EdgeClass.RITUALIZED
    if voi_is_low and cons_is_low:
        return EdgeClass.RARE
    return EdgeClass.MIDDLE


def _build_visits_df(
    rows: list[dict[str, Any]], has_embeddings: bool
) -> pd.DataFrame:
    if not has_embeddings or not rows:
        return pd.DataFrame(
            {
                "visit_seq": pd.Series(dtype=np.int64),
                "trajectory_id": pd.Series(dtype=object),
                "timestep": pd.Series(dtype=np.int32),
                "node_id": pd.Series(dtype=object),
                "embedding": pd.Series(dtype=object),
            }
        )
    df = pd.DataFrame(rows)
    df["visit_seq"] = df["visit_seq"].astype(np.int64)
    df["timestep"] = df["timestep"].astype(np.int32)
    return df
