# Stage 4b — MMLU cross-benchmark replication: pre-design notes

**Status**: pre-registered (2026-05-05). Revised post-smoke same day.
Written before the full-N experiment runs.
**Predecessor**: stage-4a replication at N=1273 on MedQA-USMLE (writeup
``docs/exploration/2026-05-05-stage-4a-replication-n1273.md``).
**Scope**: this experiment tests **per-trajectory-component
generalization across professional domains in single-trajectory-per-
question MCQ format**. It is NOT a framework-architecture validation;
the recovered-graph components (``voi_flatness``,
``distance_from_trajectory`` in its multi-trajectory aggregation form)
are structurally disabled by the single-trajectory-per-question
shape, just as they were on MedQA. Architectural-test predictions
P1-P4 from stage-6 cannot be tested here — multi-trajectory
aggregation requires multi-trajectory data. What this experiment can
establish: whether ``mean_entropy`` (the empirically dominant
per-component scorer at MedQA) generalises to non-medical
professional reasoning at full N.

---

## 0. Pre-registration revision (2026-05-05, post-smoke)

The first version of these notes (committed earlier 2026-05-05)
carried over the stage-4a corrected-3 composite as the primary
scorer. The N=20 smoke run plus a targeted composite-construction
sweep on existing smoke data revealed three findings that motivated
this revision before the full-N run.

### Finding 1 — mean_entropy alone dominates all composite constructions

Across 5 smoke subjects (N=20 each), sign-aware AUC means:
- ``mean_entropy`` alone: **0.832** (5/5 domains direction "greater")
- corrected-3 composite: 0.677 (3/5 "greater", 2/5 "less")
- mass-flipped composite: 0.674 (3/5 "greater")
- original-3 composite: 0.629 (3/5 "greater")
- all-flipped composite: 0.679 (mathematical mirror of corrected-3
  under sign-aware reporting)

``mean_entropy`` is the only construction where every domain holds
the framework's higher-defers convention. Every composite flips
direction on 2 of 5 domains, with *different* domains flipping for
each construction — the signature of small-N noise on rank-percentile
aggregation of variable-direction-stability components.

### Finding 2 — Mass-capture sign-correction hypothesis tested and not supported

A specific PI hypothesis (mass_capture_mean shows "less" direction on
4/5 smoke domains, suggesting the framework's default direction is
inverted) was tested by constructing a 5-component composite with
mass_capture inverted. Mass-flipped mean AUC = 0.674; corrected-3
mean AUC = 0.677. The targeted flip does not improve the composite.

The "4/5 less direction" pattern on smoke initially looked like a
real per-component finding, but Finding 4 (below — MedQA N=1273
shape analysis) shows this was AUC-ranking artifact driven by
per-domain outliers at N=20. At MedQA scale the pattern collapses to
near-zero central-tendency separation. The mass-capture story is
substantially revised in P5 below.

### Finding 3 — Decomposer regression on MMLU prompts (already fixed)

The strict answer-extraction regex
``^Final answer:\s*([A-D])\s*$`` rejected 35-60% of qwen2.5:7b
outputs on MMLU prompts as "no final answer extracted." This affected
Conditions A and B's ``predicted_answer`` field but NOT Condition C
(token-probability argmax bypasses the decomposer). Fixed in commit
``39ffb0c``: the regex now handles markdown emphasis, trailing
punctuation, parenthesized letters, and selects the LAST match for
self-correction patterns. 13 new tests pin the deviation classes.
The fix does not change Condition C's signatures — the analytical
findings above are computed against Condition C's clean predictions
and are unaffected.

### Finding 4 — Mass-capture shape analysis on MedQA N=1273 reveals smoke "less direction" was small-N artifact

A targeted shape analysis on the existing stage-4a MedQA replication
(N=1273) tested whether the smoke 4/5 "less direction" pattern on
mass_capture_mean reflected real central-tendency separation between
correct and wrong trajectories, or AUC-ranking artifact driven by
small-N outliers.

Plots saved at ``~/work/eunosia/artifacts/medqa-stage-4a-n1273-shape-analysis/``
(histogram, ECDF, per-position trajectory ribbons).

Stratified central-tendency results on MedQA N=1273:

| Statistic | mass_capture_mean | mass_capture_min |
|---|---|---|
| Δ_mean (wrong − correct) | -0.003 | **+0.009** |
| Δ_median | -0.001 | **+0.009** |
| Cohen's d | -0.093 | +0.107 |
| Mann-Whitney p | 0.374 | 0.022 |
| KS p | 0.688 | 0.018 |

For mass_capture_mean: distributions are statistically
indistinguishable. Cohen's d ≈ 0.09 (negligible). The
"low capture = uncommitted = boundary" mechanism is not supported
at scale.

For mass_capture_min: tiny effect (d ≈ 0.11) in the **opposite
direction** from the prediction — wrong trajectories have *slightly
higher* min capture. p-values < 0.05 only because N=1273 amplifies
trivial differences.

Per-position dynamics (real but micro-scale, all |Δ_median| < 0.01):

| pos | med_correct | med_wrong | Δ_median | MW p |
|---|---|---|---|---|
| 0 (prior) | 0.828 | 0.837 | +0.009 | 0.013 |
| 1 | 0.960 | 0.954 | -0.006 | 0.001 |
| 2 | 0.949 | 0.939 | -0.010 | 0.013 |
| 3 | 0.966 | 0.956 | -0.009 | 0.001 |
| 4-5 | converged | converged | ≈ 0 | n.s. |

Small but real pattern: wrong trajectories enter with *slightly higher*
prior commitment, then correct trajectories pull ahead during reasoning
steps, with crossover by position 4-5. Effect sizes universally
negligible.

Tail concentration (informative test):

- mass_capture_mean bottom decile (≤0.877): 128 trajectories,
  48.4% wrong vs 39.5% base rate → **+8.9pp lift** (1.22× enrichment).
  Modest but meaningful tail signal.
- mass_capture_min bottom decile (≤0.708): 128 trajectories,
  36.7% wrong vs 39.5% base rate → -2.8pp (no signal).

**Conclusion:** the smoke 4/5 "less direction" pattern was AUC-ranking
artifact driven by per-domain outliers at N=20. At N=1273 mass capture
has no central-tendency separation, has tiny opposite-direction effect
on min capture, but does show a modest bottom-decile concentration of
wrong cases on mean capture (+8.9pp lift). The mass-capture-as-
commitment-detection mechanism we hypothesized produces no meaningful
per-component deferral signal at scale on single-trajectory MCQ.

This finding directly informs P5 below — the prediction shifts from
"replicate the smoke 'less direction' pattern" to "predict the MedQA
calibrated null with mild tail concentration."

### Discipline-respecting framing of the revision

The original pre-registration was the right call given the evidence
available. The smoke surfaced empirical evidence — sufficient
across-domain consistency to justify revising the primary scorer
*before* the full-N run, rather than after. The alternative
(running full-N on corrected-3, then post-hoc reporting mean_entropy
as the better scorer) would be the discipline failure: selection on
post-experiment data. Pre-registering the revision now, on the basis
of replicated cross-domain smoke evidence plus the existing N=1273
MedQA result, is the *calibrated-claims* discipline working.

The methods paper's eventual narrative includes this discovery
sequence transparently: stage-4a corrected-3 composite was
pre-registered and replicated; stage-4b smoke evidence revealed
``mean_entropy`` as the empirically stronger per-component scorer
across cross-domain MCQ data; pre-registration revised before
stage-4b's full-N run.

---

## 1. Revised primary run scope

**Primary confirmatory run**: ``professional_law`` at N=1534. Tightest
CIs in the smoke set (lowest accuracy, n_w ~ 750 expected) and
non-medical domain — the cleanest single test of mean_entropy
cross-domain generalisation. ~2-3 hours wall on M4 Pro.

**Exploratory robustness slice (deferred to optional stage-4c)**:
``professional_accounting`` (N=282), ``professional_medicine``
(N=272), ``formal_logic`` (N=126), ``elementary_mathematics``
(N=378). Smoke results documented in this file's findings; full-N
runs not pre-committed at this stage. If full-law shows
``mean_entropy`` holds and you want broader cross-domain coverage,
stage-4c executes the remaining four subjects in a single ~2-hour
batch. If full-law shows ``mean_entropy`` does not generalise, the
remaining subjects' value is reduced and stage-4c is moot.

**Compute envelope**: same model and protocol as stage-4a
(qwen2.5:7b-instruct via ``LlamaCppLLMAdapter``). Single unattended
run.

---

## 2. Revised confirmatory predictions (professional_law, N=1534)

Seven sub-predictions tested on ``professional_law`` (P1, P2, P3, P4,
P5a, P5b, P6). Bonferroni correction at family-wise α=0.05 with 7
tests gives per-test α=0.0071 — more permissive than the original
15-test design. Results table reports both uncorrected and corrected
p-values.

All AUC predictions in **sign-aware** form (per CLAUDE.md §15
resolution; max(roc_auc, 1-roc_auc) with direction column).

### P1 — mean_entropy is the primary deferral signal

``mean_entropy`` (arithmetic mean Shannon entropy across the
trajectory's token-probability measurements) discriminates wrong-
answer trajectories on professional_law at meaningful magnitude with
the framework's convention.

> **Threshold (primary):** sign-aware AUC > 0.60 with CI lower
> bound > 0.55. Direction must be "greater".

Smoke point estimate: 0.72 [0.47, 0.92] at N=20. Full-N CI tightens
to ~±0.04 (~750 wrongs). Threshold 0.60 allows substantive shrinkage
from MedQA's 0.686 / smoke's 0.72 without calling generalisation
null.

### P2 — corrected-3 composite as exploratory baseline

The stage-4a corrected-3 composite (rank-percentile sum of
final_entropy, sign-flipped entropy_plateau, distance_from_trajectory)
is computed and reported alongside ``mean_entropy`` as an exploratory
comparison.

> **Threshold (exploratory):** mean_entropy AUC ≥ corrected-3 AUC +
> 0.05 (sign-aware). The composite is expected to underperform
> mean_entropy by at least 5 sign-aware AUC points on
> professional_law, consistent with smoke pattern.

If composite > mean_entropy at full N, that is a substantive
finding worth investigating in the writeup but does NOT replace the
methods-paper headline (mean_entropy). The composite-architecture
validation question lives at stage-6 (multi-trajectory data), not
here.

### P3 — Smoke-to-full-N consistency

``mean_entropy`` point estimate at N=1534 lands within a tight band
of the smoke estimate. Sanity check that N=20 wasn't measuring
something different from N=1534 in ways the analysis didn't catch.

> **Threshold:** |mean_entropy_full − mean_entropy_smoke| ≤ 0.10.
> Acceptable consistency band: full-N point in [0.62, 0.82]
> (smoke point 0.72 ± 0.10).
> Substantial drift: |full-N − smoke| > 0.15.

The smoke CI was [0.47, 0.92] (very wide due to N=20). Full-N point
in [0.62, 0.82] confirms smoke pattern. Full-N point in
[0.55, 0.62) or (0.82, 0.90] is mild drift, mentioned in writeup.
Full-N point < 0.55 or > 0.90 means the smoke was small-N artifact —
investigate before treating mean_entropy as a stable signal at scale.

### P4 — Per-component CI behavior on professional_law

The single-trajectory-per-question MCQ shape structurally disables
``distance_from_trajectory`` and (we expect) leaves ``entropy_plateau``
slope variable. Pre-registered prediction: their CIs at N=1534 still
include or sit close to 0.5.

> **Threshold:** ``distance_from_trajectory`` and ``entropy_plateau``
> sign-aware AUC CI lower bounds < 0.55 at N=1534 on
> professional_law. ``mean_entropy`` and ``final_entropy`` CI lower
> bounds > 0.55.

Confirms the MCQ-structural-limitation hypothesis empirically at
scale.

### P5 — Mass-capture replicates the MedQA calibrated-null pattern

Substantially revised after Finding 4 (MedQA N=1273 shape analysis).
The smoke 4/5 "less direction" pattern on mass_capture_mean was
AUC-ranking artifact driven by per-domain outliers at N=20; at MedQA
N=1273 the pattern collapses to near-zero central-tendency separation
with a modest bottom-decile concentration of wrong cases. P5 now
pre-registers replication of that calibrated-null pattern on
professional_law.

**Sub-prediction P5a — central-tendency null replicates.**

> **Threshold:** mass_capture_mean on professional_law N=1534 shows
> |Δ_median| < 0.02 between correct and wrong groups, AND
> Cohen's d < 0.15. The "low capture = uncommitted = boundary"
> mechanism produces no meaningful per-component deferral signal at
> scale.

Anchored to MedQA Δ_median = -0.001 and Cohen's d = -0.093.

**Sub-prediction P5b — bottom-decile concentration replicates.**

> **Threshold:** trajectories in the bottom decile of
> mass_capture_mean on professional_law show wrong-rate lift over
> base wrong-rate of +5pp to +15pp (anchored to MedQA's +8.9pp).
> One-sided 95% CI lower bound on lift > 0pp.

This is the *narrower* mass-capture claim that survives at scale: not
a deferral-curve signal but a tail-concentration enrichment.
Mechanism reframing: the bottom decile of mean capture identifies
trajectories where the model's reasoning across positions is
systematically less committed to letter-emission, and those
trajectories are mildly enriched for wrong answers. This is a
weaker claim than the original framework intuition but is empirically
supported.

**Falsification semantics.** P5a fails if professional_law shows
|Δ_median| ≥ 0.02 OR Cohen's d ≥ 0.15 — that would mean the
mass-capture central-tendency signal is real on professional_law but
absent on MedQA, requiring domain-specific explanation. P5b fails if
the bottom-decile lift is ≤ 0pp (CI lower bound ≤ 0) — that would
mean the modest tail concentration on MedQA doesn't generalise.

Reported as per-component finding. NOT incorporated into the
composite (smoke ``mass-flipped`` composite test did not improve
composite AUC; MedQA central-tendency null gives no reason to revisit).

### P6 — Measurement protocol robust on professional_law

The ADR-0008 token-probability measurement runs cleanly across
professional-law questions.

> **Threshold:** failure rate < 5% on professional_law (= ≤ 76
> trajectories failing of 1534).

ADR-0008 achieved zero failures on MedQA. Smoke achieved zero
condition-C failures across all 5 subjects (the decomposer issue
affected only Conditions A/B, not the measurement protocol).

---

## 3. Exploratory analyses (pre-declared)

Reported in a clearly-separated "Exploratory" section of the writeup,
NOT counted toward the methods-paper confirmatory claims.

**E1 — Composite-construction sweep at full-N.** Replicate the smoke
analysis (orig-3, corrected-3, mass-flipped, all-flipped, mean_H_only)
on professional_law at N=1534. Whether composite directions stabilise
at full N or remain noisy answers the construction-vs-noise question.

**E2 — B-vs-C complementarity at full-N.** Cell where Condition B
reports high confidence (top tertile) AND Condition C reports high
``mean_entropy`` (top tertile): wrong-rate lift over base rate. PI
flagged as potentially the most consequential cross-domain test.
Pre-specified threshold: one-sided 95% CI lower bound on lift > 0.

**E3 — Alternative entropy summaries.** Compute ``mean_entropy``,
``final_entropy``, entropy at step 0 (prior), entropy at fixed
positions; report sign-aware AUC for each. Stage-4a found
mean_entropy strongest; smoke found mean_entropy stronger than
final_entropy on 4/5 domains. Robustness check at full N.

**E4 — Mass-capture shape characterisation at full N.** Replicate the
MedQA shape analysis on professional_law: stratified central tendency
(Δ_median, Cohen's d), Mann-Whitney + KS tests on correct/wrong, ECDF
visualisation, per-position trajectory ribbons, bottom-decile lift.
Tests the P5a/P5b predictions plus characterises shape mechanism on a
non-medical domain. Per-position pattern of interest: MedQA showed
wrong group's prior-position (pos 0) median *slightly higher* than
correct, with crossover during reasoning steps (pos 1-3 correct
slightly higher), then convergence by pos 4-5. Whether this micro-
scale pattern replicates on professional_law is exploratory.

**E5 — Per-trajectory entropy curve diagnostics.** For
high-vs-low-mean_entropy trajectories, plot the entropy curve across
measurement positions. Does the wrong-class show "high entropy at all
positions" pattern (matches the stage-4a "boundary signal already at
prior" finding) or "rising entropy" or "stable high"? Informs the
mechanism story for the methods paper.

---

## 4. Smoke-test gates (exercised 2026-05-05)

Pre-registered abort conditions, plus what actually happened:

1. **Item-level failure rate > 10%** — `predicted_answer is None` on
   Conditions A (35-60%) and B (0-35%) due to decomposer regression.
   Fixed via regex relaxation (commit 39ffb0c). Condition C's
   measurement protocol was unaffected (0% failure across all 5
   subjects). Per-component AUCs computed against C's clean
   predictions.

2. **Median mass capture < 0.5** — passed (0.86-0.93 across 5
   subjects).

3. **Per-question wall time** — pending re-measurement post
   decomposer fix; smoke wall time was bounded.

4. **Component independence violated** — not directly tested at
   N=20 (correlations unstable at that N); deferred to E1
   composite-construction sweep at full-N.

Decision: **proceed to full-law run** post decomposer fix and
pre-registration revision. The smoke surfaced two real issues
(decomposer regex; composite-construction noise on small-N MCQ);
both addressed in revised pre-registration.

---

## 5. Methodology-vs-deployment-use framing

Updated for the revision:

- **Methods paper claim** (post-experiment): ``mean_entropy``
  generalises as a per-trajectory deferral signal across MCQ-format
  reasoning, replicated on professional_law at N=1534. The
  framework's composite architecture is documented as awaiting
  multi-trajectory validation at stage-6; per-component reporting
  is the cleaner claim at MCQ scale.
- **Deployment-use claim**: ``mean_entropy`` is the deployable
  scorer at present configuration. Stage-4a established it on
  MedQA-USMLE (N=1273, AUC 0.686); stage-4b extends to non-medical
  professional reasoning. Eunosia's clinical-reasoning deployment
  isn't directly informed by law performance, but the generality
  of the deployable signal informs the framework's broader
  applicability.

---

## 6. Statistical reporting policy

- **Sign-aware AUC** throughout (per CLAUDE.md §15 resolution).
- **Bootstrap CIs**: 5000 resamples per AUC, default seed 42.
- **Bonferroni correction**: 7 sub-predictions × 1 confirmatory
  subject = 7 tests; family-wise α=0.05; per-test α=0.0071.
  Reported alongside uncorrected.
- **Confirmatory vs exploratory** separation maintained throughout
  the writeup. Confirmatory: P1-P6. Exploratory: E1-E5 plus any
  post-hoc analyses motivated by the data.

---

## 7. What this experiment does not establish

(Pre-registered scope-disclaimers, reproduced in writeup.)

- **Framework-architecture validation.** Recovered-graph components
  remain structurally disabled by single-trajectory MCQ shape.
  Stage-6 chest-pain (multi-trajectory) is the architecture
  validation. This experiment cannot substitute.
- **Composite-architecture validation.** The smoke composite-
  construction sweep showed equal-weight rank-percentile
  aggregation is unstable on MCQ data; stage-6 multi-trajectory
  context is where the composite earns or fails to earn its keep.
- **Clinical-reasoning value proposition.** The deployment story
  for Eunosia is clinical chat, not professional MCQ. This
  experiment tests whether ``mean_entropy`` is domain-portable; it
  does not test whether clinical chat exhibits the signal.
- **Cross-LLM portability.** Results are for qwen2.5:7b-instruct.
  Larger models or different families may produce different
  magnitudes. PI declined cross-model 14b smokes pending the
  composite-vs-mean_entropy diagnostic; that decision can be
  revisited post-full-law if motivated.

---

## 8. Outcomes table (stub for post-experiment update)

Filled after the experiment lands. Confirms what was pre-registered
versus what was found.

| Prediction | Threshold | Result | Held? |
|---|---|---|---|
| P1 mean_entropy primary | sign-aware AUC > 0.60, CI low > 0.55, dir=greater | TBD | TBD |
| P2 corrected-3 exploratory | mean_H ≥ corrected-3 + 0.05 | TBD | TBD |
| P3 smoke→full-N consistency | \|full-smoke\| ≤ 0.10 | TBD | TBD |
| P4 per-component CIs | distance/slope CI low < 0.55; mean_H/final_H CI low > 0.55 | TBD | TBD |
| P5a mass-capture central null | mass_capture_mean \|Δ_median\| < 0.02, Cohen's d < 0.15 | TBD | TBD |
| P5b mass-capture bottom-decile lift | bottom decile of mc_mean +5 to +15pp wrong-rate lift, CI low > 0 | TBD | TBD |
| P6 protocol robust | failure rate < 5% | TBD | TBD |

Bonferroni-corrected significance per test: α=0.0071.
Family-wise interpretation reported in writeup conclusions.
