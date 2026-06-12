# Stage 4b — MMLU `professional_law` cross-benchmark replication (N=1534)

**Status**: complete (run finished 2026-05-07; analysis 2026-05-07).
**Predecessor**: stage-4a replication N=1273 on MedQA-USMLE
(``docs/exploration/2026-05-05-stage-4a-replication-n1273.md``).
**Pre-registration**: ``docs/decisions/stage_4b_mmlu_cross_benchmark_pre_design_notes.md``
(commit ``10c5e76`` for original revision; ``8c20df2`` for P5 update
post-MedQA-shape-analysis). Frozen before this run.
**Analysis script**: ``experiments/medqa_generalization/scripts/05_stage4b_analysis.py``
(commit ``8b22e84``).
**Artifact**: ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law/``.
**Analysis output**: ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law-analysis/``.

---

## Headlines

**6/7 pre-registered sub-predictions hold** on professional_law at
N=1534. The methods-paper claims survive cross-domain.

(1) ``mean_entropy`` AUC = **0.664 [0.636, 0.690]** at N=1534 on
professional_law, replicating the MedQA pattern (0.686 [0.657, 0.716])
at the framework's higher-defers convention. Cross-domain
generalisation of the per-trajectory entropy signal is empirically
confirmed.

(2) The B-vs-C complementarity finding (the methods-paper's
distinctive-contribution claim) **replicates strongly**: top-tertile
B-confidence × top-tertile mean_entropy cell shows wrong-rate 62.7%
vs base 50.1%, lift **+12.60pp [CI low +8.95pp]**. MedQA was
+16.36pp [CI low +12.34pp]. Both well above the +5pp threshold with
CIs cleanly excluding zero — Condition C's mean_entropy adds
information beyond verbalised confidence in both medical and legal
domains.

(3) The narrower mass-capture bottom-decile claim (P5b) **does not
replicate**. On MedQA, the bottom decile of mass_capture_mean had a
+8.9pp wrong-rate lift; on professional_law, the bottom decile shows
**-2.73pp** (slightly *below* base rate; CI low -9.24pp). The
mass-capture-as-tail-signal hypothesis is medical-specific.

---

## Pre-registered predictions outcomes

Bonferroni-corrected α = 0.0071 (7 sub-predictions × 1 confirmatory subject).

| Prediction | Threshold | Measured | Held? |
|---|---|---|---|
| **P1** mean_entropy primary | sign-aware AUC > 0.60, CI low > 0.55, dir=greater | AUC = 0.664 [0.636, 0.690], dir=greater | ✓ |
| **P2** corrected-3 exploratory | mean_H AUC ≥ corrected-3 + 0.05 | mean_H 0.664, corrected-3 0.593, Δ = +0.071 | ✓ |
| **P3** smoke→full-N consistency | \|full-smoke\| ≤ 0.10; band [0.62, 0.82] | smoke 0.72, full-N 0.664, Δ = 0.056 | ✓ |
| **P4** per-component CI behaviour | distance/slope CI low < 0.55; mean_H/final_H CI low > 0.55 | distance 0.49, slope 0.52, mean_H 0.64, final_H 0.60 | ✓ |
| **P5a** mass-capture central null | \|Δ_median\| < 0.02 AND Cohen's d < 0.15 | Δ_median +0.002, d +0.037, MW p = 0.595 | ✓ |
| **P5b** mass-capture bottom-decile | lift ∈ [+5pp, +15pp]; CI low > 0pp | lift -2.73pp, CI low -9.24pp | **✗** |
| **P6** measurement protocol robust | failure rate < 5% | 0/1534 = 0.00% | ✓ |

**Summary**: 6/7 hold. The single failure (P5b mass-capture
bottom-decile) is the *narrower* mass-capture claim; the *primary*
mass-capture finding (P5a central null replicates) holds with
substantially more decisive evidence than the threshold demanded.

The P5b failure is itself informative: it rules out the
mass-capture-as-tail-signal hypothesis as a domain-general
phenomenon. The MedQA +8.9pp bottom-decile lift was medical-specific.
This is the calibrated-claims discipline working — pre-registered
falsification of an exploratory hypothesis on fresh data.

---

## Methodology

Same protocol as stage-4a:

- **Model**: qwen2.5:7b-instruct via ``LlamaCppLLMAdapter`` (token-
  probability measurement, ADR-0008).
- **Embedder**: ``intfloat/multilingual-e5-large`` via
  sentence-transformers.
- **Conditions**: A (CoT-only baseline), B (CoT + verbalised
  confidence), C (CoT + token-probability per-position measurement).
- **Predicted answer per condition**: A and B from CoT-extracted
  ``Final answer:`` line (decomposer regex post-fix per
  commit ``39ffb0c`` — relaxed regex handles markdown, parens,
  trailing punctuation, picks last match for self-correction);
  C from token-probability argmax of the terminal measurement
  (per ADR-0008, decomposer-independent).
- **Wrong-answer label**: ``predicted_answer != correct_letter``.
- **Bootstrap CIs**: 5000 resamples, seed 42 (matches stage-4a).
- **Sign-aware AUC**: ``max(roc_auc, 1 - roc_auc)`` reported alongside
  direction.

Subject `professional_law` selected per pre-registration: lowest
accuracy (50.1%, 769 wrong / 765 correct of 1534) among 5 smoke
subjects, giving the largest positive class and tightest CIs at
full N. Non-medical professional reasoning, different conventions
from medicine, similar 4-option MCQ format.

Full-N wall time: ~19 hours on M1 Pro MBP at ~45s/q (per the
hardware memory; not M4 Pro). Decomposer fix held perfectly
(0/1534 failures across all three conditions).

---

## Results

### Per-component sign-aware AUCs (95% bootstrap CI)

| Scorer | sign-aware AUC | CI | direction |
|---|---|---|---|
| `mean_entropy` | **0.664** | [0.636, 0.690] | greater |
| `max_entropy` | 0.639 | [0.611, 0.666] | greater |
| `final_entropy` | 0.632 | [0.604, 0.659] | greater |
| `initial_entropy` (prior, pos 0) | 0.629 | [0.601, 0.656] | greater |
| `corrected-3` (composite) | 0.593 | [0.564, 0.621] | greater |
| `composite` (orig-3, parquet col) | 0.554 | [0.525, 0.583] | **less** |
| `entropy_plateau` | 0.551 | [0.522, 0.579] | **less** |
| `distance_from_trajectory` | 0.521 | [0.492, 0.551] | **less** |
| `mass_capture_min` | 0.511 | [0.482, 0.540] | greater |
| `mass_capture_mean` | 0.508 | [0.478, 0.537] | greater |
| `voi_flatness` | 0.500 | [0.500, 0.500] | greater (degenerate) |

The full ranking matches MedQA's pattern essentially line-for-line.
mean_entropy dominates; entropy summaries (max, final, initial)
cluster at 0.63-0.64; composites at 0.55-0.59; distance and slope
near chance; mass-capture summaries effectively at chance;
voi_flatness structurally degenerate (single-trajectory MCQ has no
graph variance).

The composite (orig-3) appears in **"less" direction** at AUC 0.554 —
exactly the artifact pattern we identified on MedQA (where
condition_comparison.csv reported sign-flipped composite). When
sign-aware reporting is applied (per CLAUDE.md §15 resolution), the
magnitude is honest at 0.554, well below mean_entropy's 0.664.

### E1 — Composite-construction sweep at full N

| Construction | sign-aware AUC | CI | direction |
|---|---|---|---|
| `orig-3` (entropy_plateau + voi_flatness + distance) | 0.554 | [0.525, 0.583] | less |
| `corrected-3` (final_H + (-slope) + distance) | 0.593 | [0.564, 0.621] | greater |
| `mass-flipped` (corrected-3 + (1-mc_mean) + (1-mc_min)) | 0.556 | [0.527, 0.585] | greater |
| `all-flipped` | 0.593 | [0.564, 0.621] | less |
| **`mean_H_only`** | **0.664** | **[0.636, 0.690]** | **greater** |

Replicates the smoke + MedQA pattern exactly:
- **mean_H_only dominates** by Δ +0.07-0.11 sign-aware AUC over every
  composite construction.
- **Mass-flipped does NOT improve over corrected-3** (0.556 vs 0.593;
  in fact slightly worse). The mass-capture-sign-correction hypothesis
  is rejected at full N on a non-medical domain — confirms the smoke
  finding and the MedQA shape analysis.
- **all-flipped ≈ corrected-3** (mathematical identity under
  sign-aware reporting).

The composite-architecture story is now empirically clear at MCQ
scale: equal-weight rank-percentile aggregation is unstable, dominated
by mean_entropy, and mass-capture inversion does not rescue it.
Stage-6 multi-trajectory data is where the composite gets its real
test.

### E2 — B-vs-C complementarity (methods-paper distinctive contribution)

The methods-paper's distinctive-contribution claim: where Condition B
reports high confidence AND Condition C reports high mean_entropy,
the wrong-rate exceeds base rate.

| | value |
|---|---|
| Top-tertile B confidence threshold | (top tertile of B's verbalised confidence) |
| Top-tertile mean_entropy threshold | (top tertile of mean_entropy) |
| Cell size (both top tertiles) | 491 |
| Cell wrong-rate | **62.7%** |
| Base wrong-rate | 50.1% |
| Lift over base | **+12.60pp** |
| One-sided 95% CI low on lift | **+8.95pp** |
| Holds (≥+5pp, CI low > 0)? | **YES** |

**Cross-domain replication compared to MedQA**:

| | MedQA N=1273 | professional_law N=1534 |
|---|---|---|
| Cell size | 417 | 491 |
| Cell wrong-rate | 55.9% | 62.7% |
| Base wrong-rate | 39.5% | 50.1% |
| Lift over base | +16.36pp | **+12.60pp** |
| CI low | +12.34pp | +8.95pp |

The lift attenuates from MedQA's +16.36pp to law's +12.60pp — about
a quarter smaller in magnitude — but remains large, statistically
clean, and well above the +5pp pre-registered threshold. The
direction matches.

This is the most consequential cross-domain finding: the framework
adds discriminative information specifically in the cell where
verbalised confidence reports "I'm confident" but mean_entropy
reports "trajectory-level uncertainty is elevated". This replicates
across two structurally different reasoning domains (clinical
diagnosis on USMLE vs legal reasoning on MMLU).

### E3 — Alternative entropy summaries

| Summary | sign-aware AUC | CI |
|---|---|---|
| `mean_entropy` | **0.664** | [0.636, 0.690] |
| `max_entropy` | 0.639 | [0.611, 0.666] |
| `final_entropy` | 0.632 | [0.604, 0.659] |
| `initial_entropy` (prior, pos 0) | 0.629 | [0.601, 0.656] |

mean_entropy dominates as expected; max_entropy is the runner-up
(slightly above final_entropy and initial_entropy, both of which sit
together in the 0.629-0.632 range). The narrow 0.629-0.664 range
across summaries shows the entropy signal is robust — multiple
operationalisations all detect it, with mean integration doing
modestly better than any single position.

### E4 — Mass-capture shape characterisation

Stratified central-tendency results:

| Statistic | mass_capture_mean | mass_capture_min |
|---|---|---|
| Δ_mean (wrong − correct) | +0.0018 | +0.0082 |
| Δ_median | +0.0022 | +0.0053 |
| Cohen's d | +0.037 | +0.057 |
| Mann-Whitney p | 0.595 | 0.467 |
| KS p | 0.628 | 0.423 |

Distributions statistically indistinguishable. Cohen's d
substantially smaller than the MedQA value (+0.037 vs MedQA's -0.093
on mc_mean) — at full N=1534 the mass-capture central-tendency
signal is even more clearly null than on MedQA.

**Both mass_capture_mean and mass_capture_min show wrong-class
slightly *higher* than correct-class** (positive Δ on both). This is
the same direction as MedQA's mass_capture_min finding — wrong cases
have minimally higher commitment, opposite to the framework's
original "uncommitted = boundary" mechanism intuition. The framework's
mass-capture-as-commitment-detection mechanism is not empirically
supported as a per-component deferral signal at scale on
single-trajectory MCQ; this finding now replicates on a non-medical
domain.

The bottom-decile concentration (P5b) — MedQA's narrower surviving
claim — does NOT replicate on law. Lift is -2.73pp (slightly below
base rate), CI low -9.24pp. The methods paper should record:
mass-capture's only surviving MedQA signal (bottom-decile lift) is
medical-specific.

### E5 — Per-position trajectory diagnostics

Per-position median entropy and mass_capture, stratified by
correct/wrong:

| Position | n_c | n_w | h_med_c | h_med_w | **Δ_med (h)** | mc_med_c | mc_med_w | Δ_med (mc) |
|---|---|---|---|---|---|---|---|---|
| 0 (prior) | 765 | 769 | 0.790 | 1.145 | **+0.354** | 0.665 | 0.665 | -0.001 |
| 1 | 765 | 769 | 0.361 | 0.627 | **+0.266** | 0.965 | 0.965 | -0.000 |
| 2 | 765 | 769 | 0.158 | 0.414 | **+0.256** | 0.961 | 0.962 | +0.001 |
| 3 | 765 | 766 | 0.088 | 0.264 | **+0.176** | 0.951 | 0.951 | +0.000 |
| 4 | 730 | 729 | 0.050 | 0.196 | **+0.146** | 0.952 | 0.953 | +0.000 |
| 5 | 534 | 527 | 0.024 | 0.072 | **+0.048** | 0.967 | 0.964 | -0.003 |

**Boundary signal already at prior** — replicating the stage-4a
finding decisively on professional_law. Wrong-class entropy is
**+0.354 bits higher than correct-class at position 0** (the prior,
before any reasoning has happened). The model's prior over the
answer space already reflects whether the question is hard. Reasoning
preserves and refines this signal but does not create it: the gap
narrows monotonically from +0.354 → +0.048 across reasoning steps
0-5, with both classes converging toward zero entropy by step 5.

This is consistent with M11's continuous-thermometer framing: the
boundary signal is detectable continuously across reasoning,
strongest at the entry point. It also explains why mean_entropy
dominates max_entropy and final_entropy — averaging across the
entire trajectory captures the prior signal where it's strongest,
whereas final_entropy attenuates as reasoning resolves.

Mass capture per-position remains essentially flat (|Δ_med| ≤ 0.003
across all positions) — confirms the central-tendency null with
position-resolved data.

Plots saved at
``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law-analysis/``:

- ``plot_mass_capture_shape.png``: histograms + ECDFs of
  mass_capture_mean and mass_capture_min by correct/wrong.
- ``plot_per_position_trajectories.png``: per-position median + IQR
  of entropy and mass_capture, by correct/wrong, showing the
  prior-position entropy gap and convergence.

---

## Methodology vs deployment use

Per the M12 distinction:

**Methods-paper claim** (post-experiment): mean_entropy generalises
as a per-trajectory deferral signal across MCQ-format reasoning
beyond medical, with cross-domain magnitude AUC 0.664 (law) vs 0.686
(medicine) — modest attenuation, same direction, comparable CI
width. The framework's distinctive contribution (B-vs-C
complementarity, +12.60pp lift on law vs +16.36pp on medicine)
replicates strongly. The mass-capture-as-commitment-detection
mechanism does not survive at scale; the methods paper updates that
section to record the negative finding honestly.

**Deployment-use claim**: ``mean_entropy`` thresholding remains the
empirically-validated deferral signal for Eunosia Phase 1. Stage-4b
extends the deployable signal's empirical validation beyond medical
reasoning at MCQ format. Whether clinical *chat* (multi-turn,
open-hypothesis differential) exhibits the signal is the next-stage
question — chest-pain stage 6.

---

## Discovery sequence (transparent reporting)

1. **Stage-4a (2026-05-05)**: pre-registered the corrected-3 composite
   as primary, replicated at N=1273 with AUC 0.591 [0.560, 0.622].
   Post-hoc: mean_entropy alone produces AUC 0.686 [0.657, 0.716],
   outperforming every composite tested.
2. **Stage-4b initial pre-design (2026-05-05 morning)**: carried
   over corrected-3 as primary, mean_entropy as P2 exploratory.
3. **Stage-4b smoke + composite-construction sweep (2026-05-05)**:
   revealed mean_entropy alone dominates all composite constructions
   across 5/5 MMLU domains. Mass-capture-flip hypothesis tested and
   not supported.
4. **Stage-4b pre-registration revision (commit ``10c5e76``)**:
   mean_entropy promoted to P1 primary; corrected-3 demoted to P2
   exploratory.
5. **MedQA shape analysis (2026-05-05)**: showed the smoke "4/5 less
   direction" pattern was AUC-ranking artifact at N=20. At MedQA
   N=1273: mass_capture central-tendency null, modest bottom-decile
   lift (+8.9pp).
6. **Stage-4b P5 revision (commit ``8c20df2``)**: mass_capture
   pre-registered as P5a (central null) + P5b (bottom-decile lift),
   anchored to MedQA-derived thresholds.
7. **Stage-4b full-law run (2026-05-06 → 2026-05-07)**: 6/7
   predictions hold. mean_entropy primary holds at AUC 0.664; B-vs-C
   complementarity replicates at +12.60pp; mass-capture P5a holds
   decisively; **P5b fails** (the narrower MedQA-specific claim does
   not generalise).

The pre-registration revisions before the full-N run, plus the
explicit threshold-anchoring on MedQA-N=1273 evidence, mean the
results are honestly tested rather than selected. P5b's failure is
itself a calibrated-claims success: the framework's mass-capture
story now narrows further, with the methods paper recording the
empirical finding rather than the hypothesised one.

---

## What this experiment establishes / does not establish

**Establishes**:
- mean_entropy's cross-domain generalisation from medical reasoning
  to non-medical professional reasoning, at MCQ format, at full N.
  AUC 0.664 [0.636, 0.690] on law vs 0.686 [0.657, 0.716] on
  medicine.
- The framework's measurement protocol (ADR-0008) is robust across
  domains (P6: 0/1534 failures).
- The framework's distinctive-contribution claim (B-vs-C
  complementarity) replicates beyond medicine: +12.60pp lift on law
  vs +16.36pp on medicine, both with CI excluding zero.
- The boundary-signal-at-prior pattern (entropy at position 0
  separates correct from wrong even before reasoning happens)
  replicates: +0.354 bits Δ_median at prior, decaying through
  reasoning steps. M11 continuous-thermometer framing supported
  empirically on a second domain.

**Does not establish**:
- **Framework-architecture validation.** Recovered-graph components
  (``voi_flatness``, ``distance_from_trajectory`` in multi-trajectory
  aggregation form) remain structurally disabled by single-trajectory
  MCQ shape. Stage-6 chest-pain (multi-trajectory self-consistency)
  is the architecture-validation experiment.
- **Composite-architecture validation.** The composite-construction
  sweep at full N confirms equal-weight rank-percentile aggregation
  is unstable on MCQ data. mean_H_only dominates by ~7-11 sign-aware
  AUC points. Stage-6 multi-trajectory context is where composites
  earn or fail to earn their keep.
- **Clinical-reasoning value proposition.** Eunosia Phase 1
  deployment is clinical chat, not professional MCQ. This experiment
  tests whether mean_entropy is domain-portable across MCQ; it does
  not test whether clinical chat exhibits the signal.
- **Cross-LLM portability.** Results are for qwen2.5:7b-instruct on
  M1 Pro. Larger models or different families may produce different
  magnitudes.
- **Mass-capture's commitment-detection mechanism.** The MedQA
  bottom-decile lift (+8.9pp) does NOT replicate on law (-2.73pp).
  The mass-capture story is now: (a) central-tendency null
  replicates cross-domain; (b) MedQA's narrower bottom-decile claim
  was medical-specific. The methods paper records this honestly
  rather than overclaiming.

---

## Next steps

**Immediate (this week)**:
- Update CLAUDE.md §15 cross-benchmark question to mark resolved
  (cross-benchmark replication landed; mean_entropy generalises;
  composite-architecture awaits stage-6).
- Update memory entries to reflect the new evidence: mean_entropy
  is cross-domain on MCQ; mass-capture story narrows further to a
  pure central-tendency null with no surviving secondary claim.

**Stage-4c (optional)**: extend cross-domain coverage to remaining
four MMLU subjects (accounting, medicine, formal_logic,
elementary_mathematics). Total ~1100 questions, ~14 hours wall on
M1 Pro. Adds robustness if further cross-domain spread is wanted
for the methods paper.

**Stage-6 chest-pain pre-registration finalisation**: integrate
Phase-C semantic-entropy predictions (P1/P2/P3 from
``stage_6_mlx_adapter_pre_design_notes.md``) into the existing
stage-6 pre-design notes. Bonferroni recomputed across the
integrated stage-6 prediction family.

**Phase A MLX adapter implementation**: now unblocked. Cross-adapter
agreement test (§7.1 of MLX adapter spec) certifies both adapters
for measurement use; framework remains adapter-agnostic.

---

## Pointers

- **Pre-registration**: ``docs/decisions/stage_4b_mmlu_cross_benchmark_pre_design_notes.md``
- **Analysis script**: ``experiments/medqa_generalization/scripts/05_stage4b_analysis.py``
- **Stage-4a writeup (predecessor)**: ``docs/exploration/2026-05-05-stage-4a-replication-n1273.md``
- **MedQA shape analysis (anchor for P5)**: artifacts at
  ``~/work/eunosia/artifacts/medqa-stage-4a-n1273-shape-analysis/``
- **Smoke artifacts (anchors for E1, E3)**: artifacts at
  ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-smoke-{subject}/``
- **Full-law artifacts**: ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law/``
- **Analysis outputs**: ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law-analysis/``
