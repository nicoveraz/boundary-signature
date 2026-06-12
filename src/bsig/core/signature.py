"""Structural signature components for boundary detection.

Three pure functions over (Trajectory, AssemblyGraph, FAISS indices)
plus a composite that combines them via rank-percentile normalization.

Components (raw, returned as-is from per-component functions):

- ``mean_entropy(trajectory) -> float``: arithmetic mean of
  hypothesis-distribution Shannon entropy across the trajectory's
  measurement positions. Higher = the model carried high uncertainty
  throughout reasoning; lower = the model concentrated on a single
  hypothesis at some point. Promoted to first-class status after the
  N=1273 stage-4a replication (2026-05-05) reported AUC 0.686
  [0.657, 0.716] on MedQA-USMLE — the strongest single-component
  signal at this configuration. Distinct from ``entropy_plateau``,
  which measures the *slope* of entropy across timesteps; mean_entropy
  measures the *level*. Empirically, level outperforms slope on
  closed-MCQ-format reasoning where the boundary signal is present
  from the prior measurement (step 0) and reasoning preserves rather
  than creates the signal.

- ``entropy_plateau(trajectory) -> float``: signed slope of
  hypothesis-distribution entropy across the trajectory's timesteps,
  fitted via least squares. Negative slope = entropy decreasing
  (converging); near-zero = plateau (model has stalled); positive
  slope = entropy increasing (model gaining uncertainty over time —
  itself a boundary signal). Sign preserved; squashing with abs()
  would discard the direction.

- ``voi_flatness(trajectory, graph) -> float``: mean ``|VoI|`` across
  the trajectory's edges. Each edge's VoI is looked up in the
  recovered ``graph``. Out-of-distribution edges (the trajectory
  took an action no historical trajectory took at this state) are
  proxied with the maximum graph VoI — they contribute to the
  "non-flat" reading, distinguishing them from in-distribution low-VoI
  steps. ``abs()`` is applied per-edge before averaging because
  recovery can produce negative VoI values (action increased posterior
  entropy on small samples); negative values indicate signal, not
  noise. Low value = consistently uninformative actions (stuck-in-loop
  signal); high value = informative actions or OOD edges.

- ``distance_from_trajectory(trajectory, visits, embedding_indices, k=5)
  -> float``: max across trajectory timesteps of the mean cosine
  distance to the k nearest historical state visits at that timestep.
  Cosine via FAISS ``IndexFlatIP`` requires L2-normalized embeddings;
  the function asserts both preconditions on first lookup. Max-aggregation
  captures the "weakest link in the chain" semantics — a trajectory
  with one weird state still flags as boundary.

Composite:

- ``compute_signatures(...)`` produces a per-trajectory DataFrame with
  the three raw components plus a composite. The composite is a
  weighted convex combination of the three components after each is
  normalized to its empirical rank-percentile in the input batch.
  Rank-percentile is dataset-relative: composite scores are NOT
  directly comparable across datasets without re-normalization. The
  ``signature_metadata.json`` artifact records ``"normalization":
  "rank_percentile"`` so downstream analyses can detect this.

Caller responsibility for trajectory / FAISS-index decoupling:

The chest-pain gate experiment must not score the same trajectories
that produced the FAISS indices. Self-distance is trivially low and
the deferral curve becomes artificially optimistic. Callers pass a
held-out test set (preferred) or implement leave-one-out by
constructing per-trajectory FAISS indices that exclude the trajectory
being scored. ``compute_signatures`` does not enforce decoupling —
it computes against whatever indices it's given. See
``experiments/chest_pain_min/`` for the holdout-split convention.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from bsig.core.graph import AssemblyGraph
from bsig.core.trajectory import State, Trajectory

if TYPE_CHECKING:
    import faiss


# ---- Public types ----


@dataclass(frozen=True, slots=True)
class SignatureWeights:
    """Convex weights for composite signature score.

    Equal-thirds default is for testing, smoke tests, and the
    ``compute_signatures`` defaults — NOT the gate-experiment
    measurement. The chest-pain gate experiment loads its own
    ``signature_weights.yml`` with values set deliberately during
    the stage 6 design pass.

    Validation: all weights non-negative; sum to 1.0 ± 1e-6.
    """

    entropy_plateau: float = 1.0 / 3.0
    voi_flatness: float = 1.0 / 3.0
    distance_from_trajectory: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        for name, val in (
            ("entropy_plateau", self.entropy_plateau),
            ("voi_flatness", self.voi_flatness),
            ("distance_from_trajectory", self.distance_from_trajectory),
        ):
            if val < 0:
                raise ValueError(f"{name} weight must be >= 0, got {val}")
        total = (
            self.entropy_plateau
            + self.voi_flatness
            + self.distance_from_trajectory
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0 ± 1e-6, got {total}")


# ---- Per-component functions ----


def mean_entropy(trajectory: Trajectory) -> float:
    """Mean Shannon entropy (bits) across the trajectory's hypothesis
    distributions.

    For each :class:`~bsig.core.trajectory.State` whose
    ``hypothesis_distribution`` is populated, computes the Shannon
    entropy of the distribution; returns the arithmetic mean across
    all such states.

    Returns ``0.0`` if no state in the trajectory has
    ``hypothesis_distribution`` populated — consistent with the
    "uninformative input → zero raw signal" convention used by the
    other per-component functions.

    Operationalises the *level* of model uncertainty across the
    reasoning trajectory (vs ``entropy_plateau``, which captures the
    *rate of change*). Empirically the strongest single-component
    deferral signal on MedQA-USMLE at the 2026-05-05 stage-4a
    replication (N=1273, AUC 0.686 [0.657, 0.716]). Used directly as
    a deployment-side score (see :mod:`bsig.core.calibration`); not
    incorporated into the default ``compute_signatures`` composite —
    that remains the pre-registered three-component aggregation.
    """
    distributions = [
        s.hypothesis_distribution
        for s in trajectory.states
        if s.hypothesis_distribution is not None
    ]
    if not distributions:
        return 0.0
    entropies = [_dict_entropy(d) for d in distributions]
    return float(np.mean(entropies))


def entropy_plateau(trajectory: Trajectory) -> float:
    """Signed slope of hypothesis-distribution entropy vs timestep.

    Returns 0.0 if fewer than 2 states have ``hypothesis_distribution``
    populated (slope is undefined for <2 points).
    """
    distributions = [
        s.hypothesis_distribution
        for s in trajectory.states
        if s.hypothesis_distribution is not None
    ]
    if len(distributions) < 2:
        return 0.0
    timesteps = np.arange(len(distributions), dtype=np.float64)
    entropies = np.array(
        [_dict_entropy(d) for d in distributions], dtype=np.float64
    )
    slope, _ = np.polyfit(timesteps, entropies, 1)
    return float(slope)


def voi_flatness(trajectory: Trajectory, graph: AssemblyGraph) -> float:
    """Mean ``|VoI|`` across trajectory edges. OOD edges proxy with
    max graph VoI.

    Returns 0.0 if the trajectory has no actions.
    """
    if len(trajectory.actions) == 0:
        return 0.0

    max_graph_voi = _max_abs_graph_voi(graph)

    abs_vois: list[float] = []
    for i, action in enumerate(trajectory.actions):
        s = trajectory.states[i].node_id
        t = trajectory.states[i + 1].node_id
        edge_voi = _lookup_edge_voi(graph, s, action.action_id, t)
        if edge_voi is None or math.isnan(edge_voi):
            abs_vois.append(max_graph_voi)
        else:
            abs_vois.append(abs(edge_voi))

    return float(np.mean(abs_vois))


def distance_from_trajectory(
    trajectory: Trajectory,
    visits: pd.DataFrame,
    embedding_indices: Mapping[int, "faiss.Index"],
    k: int = 5,
) -> float:
    """Max k-NN cosine distance across trajectory timesteps.

    Per-timestep value is the mean cosine distance to the ``k`` nearest
    historical state visits at that timestep. Trajectory-level value
    is the maximum of the per-timestep values.

    Preconditions (checked at first lookup, raises ``ValueError`` on
    violation):
    - Each ``embedding_indices[t]`` must be a ``faiss.IndexFlatIP``.
    - State embeddings must be L2-normalized to within 1e-3 tolerance.

    Returns 0.0 if the trajectory has no embedded states or none of
    its timesteps appear in ``embedding_indices``.

    ``visits`` is accepted but currently unused by this implementation
    — the FAISS index already carries the embeddings via internal IDs.
    Reserved for diagnostic queries that map FAISS results back to
    visit metadata.
    """
    import faiss  # noqa: PLC0415

    del visits  # reserved for future diagnostic mapping

    distances: list[float] = []
    for state in trajectory.states:
        if state.embedding is None:
            continue
        ts = state.timestep
        index = embedding_indices.get(ts)
        if index is None:
            continue

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError(
                f"embedding_indices[{ts}] must be IndexFlatIP for cosine "
                f"distance via inner product; got "
                f"{type(index).__name__}"
            )
        norm = float(np.linalg.norm(state.embedding))
        if not math.isclose(norm, 1.0, rel_tol=1e-3):
            raise ValueError(
                f"State embedding at timestep {ts} not L2-normalized "
                f"(norm={norm:.6f}); IndexFlatIP requires unit-norm "
                f"embeddings for cosine semantics"
            )

        query = state.embedding.reshape(1, -1).astype(np.float32)
        k_use = min(k, index.ntotal)
        if k_use == 0:
            continue
        similarities, _ = index.search(query, k_use)
        mean_sim = float(np.mean(similarities[0]))
        distances.append(1.0 - mean_sim)

    if not distances:
        return 0.0
    return float(max(distances))


# ---- Mass-capture components (ADR-0008) ----


def mass_capture_mean(trajectory: Trajectory) -> float:
    """Mean ``mass_capture`` across trajectory states.

    Per ADR-0008, ``State.mass_capture`` is the fraction of next-token
    mass that landed on the hypothesis space at each measurement
    position, when produced by a token-probability measurement protocol.
    The mean across the trajectory's measurements is one candidate
    aggregation — under the *multi-hypothesis* principle, this is one
    operationalisation among several to be evaluated empirically at
    stage 4.

    Returns ``1.0`` if no state in the trajectory has ``mass_capture``
    populated (the framework's signal is uninformative; no boundary
    signal). Returns ``1.0`` for trajectories produced by measurement
    protocols that don't compute mass capture (e.g., the deprecated
    verbalised-distribution path).

    Lower values indicate the model was less committed to letter-
    answering across the reasoning trajectory; higher values indicate
    fuller commitment. Caller may invert (``1 - mass_capture_mean``)
    when using as a deferral signal where higher = boundary.
    """
    captures = [
        s.mass_capture
        for s in trajectory.states
        if s.mass_capture is not None
    ]
    if not captures:
        return 1.0
    return float(np.mean(captures))


def mass_capture_min(trajectory: Trajectory) -> float:
    """Minimum ``mass_capture`` across trajectory states.

    The "extreme-tail" aggregation: any single low-mass-capture state
    produces a low value for the trajectory. Sensitive to the boundary
    pattern observed in the N=50 mass-capture investigation
    (2026-05-04), where the lowest-mass cases coincided with the
    model wanting to continue numeric reasoning rather than commit
    to a letter. Whether this pattern generalises at stage-4 N=1273
    scale is the empirical question this scorer is designed to test.

    Returns ``1.0`` if no state has ``mass_capture`` populated.
    """
    captures = [
        s.mass_capture
        for s in trajectory.states
        if s.mass_capture is not None
    ]
    if not captures:
        return 1.0
    return float(min(captures))


# ============================================================
# Phase-B uncertainty-signal scorers (ADR-0009 schema-v4)
# ============================================================
#
# Per-position pure functions over a top-K logprobs mapping
# (token → log-probability in nats). Operate on cached top-K data
# — no inference required. Each is a candidate per-trajectory
# scorer when aggregated (mean/min/max across the trajectory's
# states).


def p_max_from_top_k(top_k_logprobs: Mapping[str, float]) -> float:
    """Maximum probability across the top-K tokens.

    Approximates "peak sharpness." High value (close to 1.0)
    indicates the model concentrated probability on a single
    token; low value indicates spread across multiple
    candidates. Computes ``exp(max(logprobs))`` over the
    top-K. For full vocabulary p_max, top-K must be large
    enough that the top-1 is included (always true for top-K
    ≥ 1).

    Returns 0.0 for an empty mapping.
    """
    if not top_k_logprobs:
        return 0.0
    max_lp = max(top_k_logprobs.values())
    return float(math.exp(max_lp))


def entropy_full_from_top_k(
    top_k_logprobs: Mapping[str, float],
) -> float:
    """Approximate Shannon entropy (in nats) over the full
    vocabulary, computed from the top-K logprobs.

    Returns ``-Σ p_i log p_i`` over the top-K, plus a residual
    term lumping the missing mass into one virtual token.
    Approximation is downward-biased when probability is
    spread across long tails; documented in ADR-0009.

    For exact full-vocabulary entropy, full-vocab measurements
    are required (vllm-mlx engine mode or equivalent).

    Returns 0.0 for an empty mapping.
    """
    if not top_k_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in top_k_logprobs.values()]
    top_k_mass_value = sum(probs)
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    residual = max(0.0, 1.0 - top_k_mass_value)
    if residual > 1e-9:
        h -= residual * math.log(residual)
    return h


def top_k_mass_from_top_k(
    top_k_logprobs: Mapping[str, float],
    k: int = 10,
) -> float:
    """Cumulative probability of the top-K (default 10) tokens.

    High value → concentrated; low → diffuse. ``k=10`` is the
    default; ``k=1`` reduces to ``p_max``.

    Returns 0.0 for an empty mapping. ``k`` is clamped to the
    available size so callers don't need to handle short
    mappings specially.
    """
    if not top_k_logprobs:
        return 0.0
    sorted_lps = sorted(top_k_logprobs.values(), reverse=True)
    use_k = min(k, len(sorted_lps))
    return float(sum(math.exp(lp) for lp in sorted_lps[:use_k]))


def gap_top2_from_top_k(top_k_logprobs: Mapping[str, float]) -> float:
    """Probability gap between top-1 and top-2 tokens.

    Large gap → decisive (single dominant candidate); small
    gap → competition between two candidates. The "two
    competing hypotheses" boundary signal.

    Returns 0.0 for an empty or single-element mapping.
    """
    if len(top_k_logprobs) < 2:
        return 0.0
    sorted_lps = sorted(top_k_logprobs.values(), reverse=True)
    return float(math.exp(sorted_lps[0]) - math.exp(sorted_lps[1]))


def gap_top1_top_k_from_top_k(
    top_k_logprobs: Mapping[str, float],
    k: int = 10,
) -> float:
    """Probability gap between top-1 and the K-th token (default
    K=10).

    Captures broader competition than ``gap_top2``: small gap
    means many candidates are close in probability to the
    leader, indicating diffuse decision; large gap means clear
    leader.

    Returns 0.0 for fewer than ``k`` tokens (gap is then
    undefined; callers can fall back to ``gap_top2`` or skip).
    """
    if len(top_k_logprobs) < k:
        return 0.0
    sorted_lps = sorted(top_k_logprobs.values(), reverse=True)
    return float(math.exp(sorted_lps[0]) - math.exp(sorted_lps[k - 1]))


# ---- Trajectory-level aggregators for uncertainty scorers ----


def _aggregate_per_state(
    trajectory: Trajectory,
    extractor: Callable[[State], float | None],
    *,
    aggregator: str = "mean",
    default: float = 0.0,
) -> float:
    """Generic per-state aggregator. ``extractor`` returns a
    scalar from each state's data, or ``None`` if missing;
    ``aggregator`` is one of ``"mean"``, ``"min"``, ``"max"``.
    Returns ``default`` if no states yield non-None values.
    """
    raw = [extractor(s) for s in trajectory.states]
    values: list[float] = [v for v in raw if v is not None]
    if not values:
        return default
    if aggregator == "mean":
        return float(np.mean(values))
    if aggregator == "min":
        return float(min(values))
    if aggregator == "max":
        return float(max(values))
    raise ValueError(
        f"aggregator must be 'mean', 'min', or 'max'; got {aggregator!r}"
    )


def mean_p_max(trajectory: Trajectory) -> float:
    """Mean ``p_max`` across the trajectory's measurement positions."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            p_max_from_top_k(s.top_k_logprobs)
            if s.top_k_logprobs
            else None
        ),
        aggregator="mean",
        default=0.0,
    )


def min_p_max(trajectory: Trajectory) -> float:
    """Minimum ``p_max`` across the trajectory."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            p_max_from_top_k(s.top_k_logprobs)
            if s.top_k_logprobs
            else None
        ),
        aggregator="min",
        default=0.0,
    )


def mean_entropy_full(trajectory: Trajectory) -> float:
    """Mean approximate full-vocab entropy across the trajectory."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            entropy_full_from_top_k(s.top_k_logprobs)
            if s.top_k_logprobs
            else None
        ),
        aggregator="mean",
        default=0.0,
    )


def mean_top_k_mass(trajectory: Trajectory, k: int = 10) -> float:
    """Mean top-K mass across the trajectory."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            top_k_mass_from_top_k(s.top_k_logprobs, k=k)
            if s.top_k_logprobs
            else None
        ),
        aggregator="mean",
        default=0.0,
    )


def min_top_k_mass(trajectory: Trajectory, k: int = 10) -> float:
    """Minimum top-K mass across the trajectory (extreme-tail
    boundary signal)."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            top_k_mass_from_top_k(s.top_k_logprobs, k=k)
            if s.top_k_logprobs
            else None
        ),
        aggregator="min",
        default=0.0,
    )


def mean_gap_top2(trajectory: Trajectory) -> float:
    """Mean top-1-vs-top-2 gap across the trajectory."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            gap_top2_from_top_k(s.top_k_logprobs)
            if s.top_k_logprobs
            else None
        ),
        aggregator="mean",
        default=0.0,
    )


def min_gap_top2(trajectory: Trajectory) -> float:
    """Minimum top-1-vs-top-2 gap across the trajectory."""
    return _aggregate_per_state(
        trajectory,
        lambda s: (
            gap_top2_from_top_k(s.top_k_logprobs)
            if s.top_k_logprobs
            else None
        ),
        aggregator="min",
        default=0.0,
    )


# ---- Provenance assertion (corruption-handling pattern) ----

_PROVENANCE_KEYS: tuple[str, ...] = (
    "adapter_name",
    "model",
    "quantization",
    "schema_version",
)


def _check_provenance_compatible(
    trajectories: Sequence[Trajectory],
) -> None:
    """Refuse to aggregate measurements from incompatible sources.

    Scans ``State.metadata`` across the input trajectories for the
    keys in ``_PROVENANCE_KEYS``. If any key has multiple distinct
    values (meaning measurements came from different adapters,
    models, quantizations, or schema versions), raises ``ValueError``
    with the conflicting keys + values surfaced.

    Provenance keys absent from metadata are not enforced (no-op for
    that key) — the assertion is opportunistic, not mandatory. As
    scripts incrementally record provenance, the check becomes
    increasingly load-bearing.

    Override path: callers that genuinely intend to aggregate across
    incompatible sources (e.g., explicit cross-comparison studies)
    pass ``force_mix=True`` to ``compute_signatures``; the assertion
    is skipped.

    The asymmetric-failure rationale: silent miscombination
    contaminates downstream aggregates and AUCs in ways that are
    hard to diagnose; loud refusal at aggregation boundary forces
    the caller to acknowledge the mix or fix the data. See
    ``docs/decisions/corruption_registry.md`` (entry 2026-05-07
    bit-identical-weights cross-adapter precondition).
    """
    seen: dict[str, set[str]] = {k: set() for k in _PROVENANCE_KEYS}
    for traj in trajectories:
        for state in traj.states:
            for key in _PROVENANCE_KEYS:
                if key in state.metadata:
                    seen[key].add(str(state.metadata[key]))
    incompatible = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if incompatible:
        raise ValueError(
            "Provenance mismatch across input trajectories: "
            f"{incompatible}. Aggregating measurements from "
            "incompatible sources silently produces wrong "
            "composite/aggregate values. Either filter to a "
            "single-provenance subset, or pass force_mix=True if "
            "the cross-source comparison is intentional."
        )


# ---- Public composite ----


def compute_signatures(
    trajectories: Sequence[Trajectory],
    graph: AssemblyGraph,
    visits: pd.DataFrame,
    embedding_indices: Mapping[int, "faiss.Index"],
    weights: SignatureWeights,
    force_mix: bool = False,
) -> pd.DataFrame:
    """Per-trajectory signature scores.

    Returns a DataFrame with columns:
    - ``trajectory_id`` (str)
    - ``mean_entropy`` (float32, mean Shannon entropy in bits across
      the trajectory's hypothesis distributions; 0.0 when no state
      carries a distribution). Promoted to first-class status after
      the N=1273 stage-4a replication; the empirically strongest
      single-component deferral signal at that configuration.
    - ``entropy_plateau`` (float32, raw signed slope)
    - ``voi_flatness`` (float32, raw mean |VoI|)
    - ``distance_from_trajectory`` (float32, raw max k-NN distance)
    - ``mass_capture_mean`` (float32, mean mass_capture across the
      trajectory's measurements; 1.0 when no state carries mass
      capture). Per ADR-0008, candidate signal evaluated at stage 4.
    - ``mass_capture_min`` (float32, min mass_capture across the
      trajectory; 1.0 when no state carries mass capture). The
      extreme-tail companion to ``mass_capture_mean``.
    - ``composite`` (float32, weighted sum of rank-percentile-
      normalized components, in [0, 1])

    The composite continues to be the weighted combination of the
    three original components (entropy_plateau, voi_flatness,
    distance_from_trajectory) per the existing
    :class:`SignatureWeights`. ``mean_entropy`` and the mass-capture
    columns are reported alongside but NOT incorporated into the
    default composite — the methods-paper claim is the pre-registered
    three-component composite; ``mean_entropy`` is a first-class
    deployment-side scorer reported beside it. Per the *calibrated
    claims* and *measurement-protocol-as-contribution* disciplines,
    promotion of any column into the composite is a separate
    decision driven by replication evidence, not an automatic
    consequence of being computed here.

    Empty input yields an empty DataFrame with the same schema.

    See module docstring on caller's responsibility for trajectory /
    FAISS-index decoupling.

    **Provenance assertion (corruption-handling pattern).** Before
    aggregating, the function scans ``State.metadata`` across input
    trajectories for the keys ``adapter_name``, ``model``,
    ``quantization``, ``schema_version``. If any key has multiple
    distinct values, raises ``ValueError`` — silent miscombination
    of measurements from incompatible adapters/models/quantizations
    contaminates aggregates in hard-to-diagnose ways. Pass
    ``force_mix=True`` to skip this check when the cross-source
    aggregation is intentional. Provenance keys absent from
    metadata are not enforced (opportunistic check; becomes load-
    bearing as scripts populate provenance). See
    ``docs/decisions/corruption_registry.md``.
    """
    if not trajectories:
        return pd.DataFrame(
            {
                "trajectory_id": pd.Series(dtype=object),
                "mean_entropy": pd.Series(dtype=np.float32),
                "entropy_plateau": pd.Series(dtype=np.float32),
                "voi_flatness": pd.Series(dtype=np.float32),
                "distance_from_trajectory": pd.Series(dtype=np.float32),
                "mass_capture_mean": pd.Series(dtype=np.float32),
                "mass_capture_min": pd.Series(dtype=np.float32),
                "composite": pd.Series(dtype=np.float32),
            }
        )

    if not force_mix:
        _check_provenance_compatible(trajectories)

    rows: list[dict[str, object]] = []
    for traj in trajectories:
        rows.append(
            {
                "trajectory_id": traj.trajectory_id,
                "mean_entropy": mean_entropy(traj),
                "entropy_plateau": entropy_plateau(traj),
                "voi_flatness": voi_flatness(traj, graph),
                "distance_from_trajectory": distance_from_trajectory(
                    traj, visits, embedding_indices
                ),
                "mass_capture_mean": mass_capture_mean(traj),
                "mass_capture_min": mass_capture_min(traj),
            }
        )

    df = pd.DataFrame(rows)
    ep_pct = _rank_percentile(df["entropy_plateau"].to_numpy())
    vf_pct = _rank_percentile(df["voi_flatness"].to_numpy())
    dt_pct = _rank_percentile(df["distance_from_trajectory"].to_numpy())

    composite = (
        weights.entropy_plateau * ep_pct
        + weights.voi_flatness * vf_pct
        + weights.distance_from_trajectory * dt_pct
    )

    df["mean_entropy"] = df["mean_entropy"].astype(np.float32)
    df["entropy_plateau"] = df["entropy_plateau"].astype(np.float32)
    df["voi_flatness"] = df["voi_flatness"].astype(np.float32)
    df["distance_from_trajectory"] = df["distance_from_trajectory"].astype(
        np.float32
    )
    df["mass_capture_mean"] = df["mass_capture_mean"].astype(np.float32)
    df["mass_capture_min"] = df["mass_capture_min"].astype(np.float32)
    df["composite"] = composite.astype(np.float32)
    return df


# ---- Internal helpers ----


def _dict_entropy(distribution: Mapping[str, float]) -> float:
    """Shannon entropy in bits over a probability distribution dict."""
    total = 0.0
    for p in distribution.values():
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def _lookup_edge_voi(
    graph: AssemblyGraph, source: str, action: str, target: str
) -> float | None:
    """Find edge ``(source, action, target)``; return its VoI or None
    if the edge is not in the graph."""
    if not graph.has_node(source):
        return None
    for edge in graph.outgoing_edges(source):
        if edge.action_id == action and edge.target_id == target:
            return edge.voi
    return None


def _max_abs_graph_voi(graph: AssemblyGraph) -> float:
    """Max absolute VoI across all graph edges; 0.0 if graph empty
    or all edges have NaN VoI."""
    if graph.num_edges == 0:
        return 0.0
    voi_array = graph._edges["voi"].to_numpy()
    valid = voi_array[~np.isnan(voi_array)]
    if len(valid) == 0:
        return 0.0
    return float(np.max(np.abs(valid)))


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    """Convert raw values to rank-percentiles in [0, 1].

    Uses scipy.stats.rankdata with method='average' (handles ties as
    average rank). For corpora with fewer than ~30 trajectories the
    rank steps are coarse (1/N granularity) — acceptable for smoke
    tests, document-flagged for the gate experiment which uses N >> 30.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.5], dtype=np.float64)
    ranks = rankdata(values, method="average")
    return np.asarray(ranks / n, dtype=np.float64)
