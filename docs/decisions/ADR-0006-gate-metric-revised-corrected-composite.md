# ADR-0006 — Gate metric revised: corrected-3 composite, not original composite

**Date:** 2026-05-07
**Status:** ACCEPTED
**Supersedes:** none (records a revision flagged in CLAUDE.md §15 as
"deferred / record at stage 4 setup")
**Superseded by:** none
**Predecessors:**
- Project charter v0.4 §5.7 (original gate: AUC ≥ 0.65 on
  need-for-consultation labels using the framework's composite signal).
- Stage-4a pilot diagnostics 2026-05-04 (principled-redesign of the
  composite producing the *corrected-3* operationalisation).
- Stage-4a replication writeup 2026-05-05
  (`docs/exploration/2026-05-05-stage-4a-replication-n1273.md`).
- Stage-4b cross-domain replication 2026-05-07
  (`docs/exploration/2026-05-06-stage-4b-mmlu-professional_law.md`).

## Context

The project charter set the chest-pain gate metric as **AUC ≥ 0.65 on
need-for-consultation labels using the framework's composite signal**,
operationalised at v0.1.0 as the weighted sum of three rank-percentile-
normalised components: `entropy_plateau` (slope), `voi_flatness`,
`distance_from_trajectory`.

Stage-4a diagnostics on the N=100 pilot (2026-05-04) identified two
load-bearing problems with the original operationalisation:

1. **`entropy_plateau` measured as slope** is anti-signal in the
   regime that matters. Confident-and-correct trajectories should
   have low entropy throughout *with no slope*; confident-and-wrong
   trajectories often have low entropy with no slope either. The
   slope is dominated by where in the CoT the model articulates
   its hypothesis, not by the boundary signal the framework targets.
   Magnitude (`final_entropy`) is the principled replacement.
2. **`voi_flatness` is structurally dead on closed-MCQ benchmarks**
   (per ADR-0002 closeout): each MCQ question yields one trajectory,
   so the canonicalizer's hash includes question text verbatim,
   making cross-question reasoning-step collapse structurally
   impossible at any bin precision. The component contributes zero
   discriminative information at MCQ scale.

The principled-redesign composite, **corrected-3**, makes three
substitutions:

- Drop `voi_flatness` (dead on MCQ).
- Replace `entropy_plateau` slope with `final_entropy` magnitude.
- Keep `distance_from_trajectory`.

CLAUDE.md §15 flagged this revision as "still open / new from
stage-4a diagnostics" with the note: *"the gate metric should be
the corrected composite (per the principled redesign in 2026-05-04
diagnostics), not the original composite."* Recording that decision
formally is the purpose of this ADR.

## Decision

**The chest-pain gate metric is AUC ≥ 0.65 on need-for-consultation
labels using the corrected-3 composite as defined in
`bsig.core.signature.compute_signatures`** (final_entropy +
distance_from_trajectory; weights from `SignatureWeights`).

Specifically:

- **Headline gate**: corrected-3 composite AUC ≥ 0.65 on the
  held-out test fold against need-for-consultation labels.
- **Replication standard**: bootstrap 95% CI lower bound ≥ 0.50.
  The point estimate alone is insufficient; the lower CI bound must
  exclude chance.
- **Reporting alongside**: `mean_entropy` AUC and Condition B
  verbalised-confidence AUC. The corrected-3 composite is what
  survives the gate; `mean_entropy` is reported as an exploratory
  finding (per `project_mean_entropy_deployable.md`) that may
  outperform the composite; Condition B is the published-baseline
  comparison.
- **Sign-aware AUC reporting**: per the conventions adopted
  2026-05-05, the gate AUC is reported as `max(AUC, 1 - AUC)` with
  `direction` field. The corrected-3 composite's expected direction
  is "greater" (high score → defer); a result with direction "less"
  at AUC ≥ 0.65 would be a methodological alarm, not a passed gate.

**Pause threshold (unchanged from charter)**: corrected-3 composite
AUC < 0.55 → pause project for re-scoping. Between 0.55 and 0.65 →
extended analysis on subgroups + Condition B comparison; gate
provisionally not passed pending the analysis.

## Why

- **Principled redesign earns its keep at scale.** The corrected-3
  composite replicated at N=1273 MedQA (AUC 0.591, CI [0.560,
  0.622]) and at N=1534 MMLU professional_law (writeup pending
  detail review; pre-registered predictions held). Both above
  chance, both with bootstrap lower bound clearing 0.50. The
  original composite is essentially equivalent at AUC 0.599 on
  MedQA but is *theoretically dirty* (anti-signal slope, dead
  component). Using a theoretically-clean operationalisation at the
  gate matters when the gate decision is load-bearing for funding
  and project continuation.

- **Pre-registration discipline is preserved.** The corrected-3
  composite was pre-registered before the N=1273 replication. The
  gate metric being the same composite continues that discipline:
  the chest-pain gate is being predicted by an operationalisation
  that has earned its empirical keep on two benchmarks before being
  used for the gate decision.

- **The MCQ benchmarks set the floor, not the ceiling.** AUC 0.591
  on MedQA and AUC 0.664 on MMLU law are upper-bounded by the
  homogeneous-question-difficulty property of those benchmarks
  (every MCQ question is independent; no graph-structural
  components come alive). Stage-6 chest-pain has multi-trajectory-
  per-encounter structure with self-consistency sampling at varied
  temperature, which is the regime where `voi_flatness` and
  graph-structural components were originally designed for. The
  corrected-3 composite drops `voi_flatness` because it's dead on
  MCQ; whether it should be re-added at stage-6 is a stage-6
  design-pass decision (recorded as a stage-6 open question, not
  resolved by this ADR).

- **The gate threshold remains 0.65, not 0.591.** The MCQ AUC of
  0.591 is the framework's *MedQA* result. The chest-pain gate
  applies to clinical-trajectory data with multi-signal weak
  supervision, where the framework's structural components have
  more substrate to operate on. Setting the gate at the MCQ AUC
  would pre-commit the project to a result that doesn't depend
  on the framework's distinguishing claim. The 0.65 bar is harder
  than the MCQ result and reflects the additional structure
  available on clinical data.

## Consequences

- **Gate-experiment scripts** (`experiments/chest_pain_min/scripts/09_evaluate.py`)
  must report the corrected-3 composite as the headline AUC. The
  original composite can be reported alongside for transparency
  but is not the gate metric.
- **Pre-registration for chest-pain gate** must reference this ADR
  and the corrected-3 composite definition explicitly. A
  pre-registration document under `docs/decisions/` or
  `docs/exploration/` should be written before the gate experiment
  runs and not modified after.
- **The Phase 1 deliverable** reports the corrected-3 composite
  AUC as the headline gate result. The mean_entropy AUC is reported
  as the exploratory deployment-side signal (per
  `project_mean_entropy_deployable.md`).
- **Methods paper headline** is the corrected-3 composite result on
  MedQA (already drafted §5.1).

## Open follow-ups

- **Stage-6 design pass**: should `voi_flatness` be re-added to the
  composite for the chest-pain gate, given that multi-trajectory-
  per-encounter structure may make the component come alive? The
  default decision is *no, keep corrected-3* — changing the gate
  metric definition mid-flight would compromise pre-registration.
  But the question is worth being explicit about. Tracked in
  `stage_6_chest_pain_pre_design_notes.md`.
- **Alternative gate metric for deployment-side signal**: the
  Eunosia Phase-1 deferral signal may use `mean_entropy` rather
  than the composite (per `project_mean_entropy_deployable.md`).
  This is a deployment-side decision separate from the gate; the
  gate uses the corrected-3 composite per this ADR.
