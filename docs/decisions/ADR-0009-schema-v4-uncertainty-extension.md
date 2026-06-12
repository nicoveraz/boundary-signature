# ADR-0009 — Schema-v4 uncertainty-signal extension to TokenProbabilityResult

**Date:** 2026-05-07
**Status:** ACCEPTED
**Supersedes:** none
**Superseded by:** none
**Predecessors:**
- ADR-0008 (token-probability measurement protocol; introduced
  schema-v3 with full top-K logprobs preservation per the
  measurement-vs-computation principle).
**Reference design:** ``docs/decisions/stage_6_mlx_adapter_pre_design_notes.md``
§5 (Phase B uncertainty-signal extensions).

## Context

Stage-4a + stage-4b empirical work established ``mean_entropy`` as
the dominant per-trajectory deferral signal at MCQ scale and
documented the rejection of mass-capture mechanism hypotheses (§5
of methods-paper draft). The empirical position naturally raises
the question: are there other per-position scorers that add
discriminative information beyond mean entropy?

The MLX adapter pre-design (commit ``4cf2c04``) listed six
candidate single-token uncertainty signals from the broader
literature:

- ``p_max``: max prob across full vocab (peak sharpness)
- ``entropy_full``: Shannon entropy over full vocab (overall spread)
- ``top_k_mass``: cumulative prob of top-K tokens (concentration)
- ``gap_top2``: prob margin top-1 vs top-2 (decisiveness)
- ``gap_top1_topK``: prob margin top-1 vs Kth (broader competition)
- ``chosen_logprob``: logprob of the emitted token (sequence
  perplexity contributor)

These signals are pure derivations from the per-position
distribution data the framework already records. Per the
**measurement-vs-computation principle (M10)**: the cached
top-K logprobs preserve enough information at full fidelity for
these scorers to be re-derived on existing trajectories without
re-running inference. Stage-4a/4b cached data (schema-v3) carries
top-40 logprobs per measurement position; new scorers can be
computed against this on existing professionals.

## Decision

Extend ``TokenProbabilityResult`` with seven additive optional
fields, each defaulting to ``None`` for backward compatibility
with schema-v2/v3 cached trajectories:

```python
@dataclass(frozen=True, slots=True)
class TokenProbabilityResult:
    # Existing schema-v3 fields
    distribution: Mapping[str, float]
    mass_capture: float
    truncated_members: tuple[str, ...] = field(default_factory=tuple)
    top_k_logprobs: Mapping[str, float] = field(default_factory=dict)

    # NEW schema-v4 fields (additive, default None)
    p_max: float | None = None
    entropy_full: float | None = None    # nats; over top_k_logprobs
    entropy_hyp: float | None = None     # nats; over distribution
    top_k_mass: float | None = None      # cumulative prob of top-K (default top-10)
    gap_top2: float | None = None
    gap_top1_topK: float | None = None
    chosen_logprob: float | None = None
```

Schema version bumps from v3 to v4. The persistence reader (per
``bsig.medqa.trajectory_sources.serialization``) handles v3
trajectories transparently — v3 schemas load with new fields as
``None``; v4 schemas carry all fields populated by adapters that
compute them.

Adapters that can compute the new fields populate them:
``LlamaCppLLMAdapter`` and ``MLXLLMAdapter`` both have access to
top-K logprobs at measurement time and populate the per-position
fields. Adapters that cannot (e.g., Ollama, which does not expose
logprobs) leave the fields ``None``.

Per-position scorer functions are added to
``bsig.core.signature``:

```python
def p_max(top_k_logprobs: Mapping[str, float]) -> float
def entropy_full_from_top_k(top_k_logprobs: Mapping[str, float]) -> float
def top_k_mass(top_k_logprobs: Mapping[str, float], k: int = 10) -> float
def gap_top2(top_k_logprobs: Mapping[str, float]) -> float
def gap_top1_top_k(top_k_logprobs: Mapping[str, float], k: int = 10) -> float
```

These are pure functions over the cached top-K logprobs;
unit-testable without LLM calls; usable as standalone derivations
from cached schema-v3 trajectories.

Trajectory-level aggregators (``mean_p_max``, ``min_p_max``,
``final_p_max``, ``mean_top_k_mass``, ``mean_gap_top2``, etc.) are
added in the same module following the existing pattern.

## Consequences

### Positive

- New scorers are testable on cached MedQA-N=1273 + stage-4b
  professional_law-N=1534 trajectories *without re-running
  inference*. The empirical question — do any of the new signals
  add information beyond mean_entropy? — is answerable on existing
  data.
- The schema extension is additive and backward-compatible; no
  re-migration of existing artifacts is required.
- The framework's *multi-hypothesis principle (M3)* is operationally
  realised: multiple scorers coexist; evaluation reports
  per-component performance; selection is empirical.

### Negative

- ``entropy_full`` computed from top-K logprobs is an *approximation*
  of full-vocabulary entropy. The cached top-40 logprobs cover
  >99% of probability mass for typical model outputs, but the
  entropy approximation is biased downward when probability is
  spread across long tails. Documentation should be explicit:
  ``entropy_full_from_top_k`` is "entropy over the top-K plus a
  lump-residual term" rather than true full-vocabulary entropy.
  For exact values, future Phase A measurements with full-vocab
  output (vllm-mlx engine mode + tensor extraction) can populate
  the field directly.
- ``chosen_logprob`` is not stored in schema-v3 cached data. It
  requires inference-time access to the actually-emitted token,
  which the cached trajectories do not preserve at the per-step
  level. Trajectories collected under schema-v4-aware adapters
  carry it; cached schema-v3 trajectories cannot derive it
  retroactively.
- Total persistence size increases by ~5-7 floats × measurement
  positions per trajectory ≈ 200 KB at N=1273. Negligible relative
  to the existing ~9 MB top-K logprobs storage.

### Reproducibility

The schema version is recorded in the cached-trajectories metadata.
A schema-v4 reader operating on a schema-v3 artifact loads new
fields as ``None``; downstream computations short-circuit when the
field is missing rather than producing spurious values.

The extension is done at the framework's adapter Protocol level
(``bsig.adapters.llm.TokenProbabilityResult``); both
``LlamaCppLLMAdapter`` and ``MLXLLMAdapter`` are updated to populate
the new fields when computing ``get_token_probabilities``. Schema-v4
serialisation is added to
``bsig.medqa.trajectory_sources.serialization``.

## Pre-registered exploratory predictions (E1, E2, E3 from
stage-6 MLX adapter pre-design §5.5)

Tested on cached MedQA-N=1273 + stage-4b professional_law-N=1534
data:

- **E1**: ``mean_p_max`` AUC against wrong-answer indicator is in
  range [0.55, 0.75] on each domain. If yes, candidate complementary
  signal. If outside the range, framework's signal is specifically
  entropy-based, not peak-sharpness-based.
- **E2**: ``mean_top_k_mass`` (top-10) AUC underperforms
  ``mean_entropy`` (Δ AUC ≥ -0.05) AND is highly correlated with
  it (Spearman r ≥ 0.5). If correlation is low, top-K mass is
  measuring something different and warrants follow-up.
- **E3**: ``mean_gap_top2`` AUC ≥ 0.60 with stronger separation in
  the bottom decile (≥ +5pp wrong-rate lift vs base rate). Tracks
  the boundary "two competing hypotheses" intuition.

E1-E3 are *exploratory* in the methods paper; their results inform
whether any of the new signals deserves promotion to confirmatory
status for stage-6 chest-pain pre-registration. The cached-data
test is methodologically clean per M10 (no re-inference; fresh
analysis on preserved measurements).

## Implementation status

- Per-position scorer functions: scaffolded in
  ``bsig.core.signature`` (commit pending).
- Schema-v4 fields on ``TokenProbabilityResult``: pending.
- Re-derivation analysis script (E1/E2/E3 evaluation on cached
  trajectories): pending.
- ``LlamaCppLLMAdapter`` field population: pending (additive — does
  not break existing pipelines).
- ``MLXLLMAdapter`` field population: pending (Phase B work).
