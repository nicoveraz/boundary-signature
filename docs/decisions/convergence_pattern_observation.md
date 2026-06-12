# Convergence-pattern observation

**Date:** 2026-05-23
**Status:** ACCEPTED (meta-observation about the project; not a research
finding, not a code change)
**Placement:** `docs/decisions/`, NOT CLAUDE.md. Promotion deferred until
multiple *additional* instances confirm the pattern — same standard as
the corruption registry and the perturbation-UQ literature record.

## The observation

Three times in recent sessions, an independently-arrived framework
intuition turned out to match established prior work once the literature
was checked:

1. **Cross-quantization disagreement** → cross-model disagreement
   literature (`project_cross_quantization_disagreement.md`, E_quant_3).
2. **Temperature perturbation** → SPUQ / Monte Carlo Temperature
   (`perturbation_uq_literature.md` §1–5, E_perturb_1).
3. **Per-token distribution shape** → LogitScope / EPR / HaluNet /
   entropy-at-fabrication-onset (`perturbation_uq_literature.md` §6,
   E_token_1/2).

## What it implies

- The framework's intellectual position is **mainstream LLM uncertainty
  quantification**, not a niche. The signals it cares about are signals
  the field also cares about.
- The contribution claim is therefore **synthesis + clinical validation
  + discipline**, not novel methodology. This is smaller than the
  original "novel measurement infrastructure" framing. It is still real
  and still publishable — as a *synthesis-with-validation* paper, not a
  *methods-paper-introducing-a-new-technique*. Being honest about which
  kind of paper this is matters, because the writing strategy differs
  (extensive related work and careful positioning vs a technique reveal).

## The honest caveat (diagnose, don't reframe)

Convergence is **weaker evidence than it feels.** Two readings are
observationally identical:

- *Inflated:* "our intuitions independently rediscovered what experts
  found, so our intuition is excellent / validated." This is
  self-congratulation and should be resisted.
- *Deflationary (correct):* LLM uncertainty quantification is a large,
  fast-moving field. Almost any reasonable per-token or perturbation
  intuition has prior art. Finding a match three times is partly
  **selection effect** — we go looking for prior work and the field is
  crowded enough that we find it. The right inference is *against*
  novelty, not *for* the framework's brilliance.

So the load-bearing conclusion is the deflationary one: convergence
lowers the novelty claim. It does **not** license a "our instincts are
validated by the field" narrative. If anything, three matches is a
prompt to assume the next independent intuition also has prior art.

## Methods-paper writing strategy (follows from the above)

- Extensive, careful related work positioning each framework signal
  against its established family.
- **Empirical validation as the primary contribution** (cross-domain
  `mean_entropy`, B-vs-C complementarity, mass-capture honest narrowing,
  Phase-B re-derivation, stage-6 when it lands).
- The **calibrated-claims discipline** as the methodological
  contribution.
- Frame the measurement protocol as *thoughtful synthesis applied to
  clinical-reasoning validation*, not as new technique.

## Working rule for future intuitions

When a fresh framework intuition appears in this conversation pattern,
**literature-check it before extending the framework, expecting
convergence rather than novelty.** Treat a prior-art match as the
default outcome. Document the positioning (as here); defer
implementation to post-stage-6 or parallel bandwidth
(`perturbation_uq_literature.md` §4 / §6 implications are deferred, not
queued).

## Calibration-failure modes: four instances

The convergence pattern, applied *reflexively*, produces wrong framing.
Four instances in recent turns, recorded honestly because the pattern is
worth surfacing, not an embarrassment to smooth over:

1. **LM-Polygraph misread (deflationary).** I treated LM-Polygraph (a
   *toolkit* — a collection of UE methods under one interface) as if it
   collapsed boundary-signature's diagnostic-framework claim. The
   convergence pattern fired without making the architectural distinction
   between a method-collection library and a diagnostic framework. The
   user correctly flagged these are architecturally different things.

2. **ConfiDx + Anatomy overcorrection (deflationary).** Having been
   burned on (1), I overcorrected and treated ConfiDx as *displacing*
   boundary-signature's contribution. But ConfiDx is a fine-tuned model
   (model-layer) and boundary-signature is a measurement protocol
   (measurement-layer) — different layers, complementary. Reading the
   *Anatomy of Uncertainty* paper carefully (rather than pattern-matching
   its title) then revealed its decomposition is by **causal source**
   (input/knowledge/decoding) while boundary-signature's is by
   **measurement signal** — different levels; and Anatomy's reasoning
   results were weak (GSM8K AUROC 0.33–0.60), making the
   clinical-reasoning validation genuinely additive.

3. **Compute-constraint omission (the deepest).** Across the *entire*
   literature engagement I missed that the framework's defining
   differentiator from the cited prior art is the **deployment-constraint
   regime** it targets (Apple Silicon, 4-bit quantized, single-run, no
   fine-tuning, no cloud, single-person team — see
   `compute_constraint_orientation.md`). Every cited diagnostic or
   model-layer solution assumes infrastructure (multi-sample, ensembles,
   fine-tuning, RAG, H100) unavailable in that regime. I narrowed the
   contribution around *secondary* features (measurement-layer
   architecture, single-run operationalization) without naming the
   constraint that motivates them. This is not a misread of one paper; it
   is a failure to characterize the framework's own ground before
   comparing it to anything.

4. **GT-target selection by data availability, not signal-target
   alignment.** The stage-6 GT was disposition (admission/discharge),
   chosen because MIMIC-IV-ED hands those labels over for free. But the
   framework's signals measure *model-output uncertainty*, so the aligned
   target is *model-output reliability* (diagnostic accuracy) — exactly
   what stage-4 used (MCQ correctness). Validating clinical UQ against
   patient disposition instead of LLM-output correctness was a
   target-type category error, masked because pre-registration ensured the
   prediction was *tested* but not that it was tested against the *aligned*
   target. (Surfaced via the GT-confound resolution;
   `stage_6_chest_pain_pre_design_notes.md` §"GT reframe".)

   **Honest status of this one:** it emerged *from* a negative result, so
   it carries rationalization risk. Two guards that it's substantive: (i)
   the aligned target (output correctness) is what stage-4 used and what
   UQ literature uses — it would have been more defensible at lock time
   *independent* of the disposition result; (ii) it does **not** rescue
   the framework — the aligned GT (diagnostic accuracy vs discharge ICD)
   *recurs* the same admission-availability skew (54% coverage, 30% of
   HOME), so the refinement names a real lesson without conveniently
   dissolving the data limitation. A purely motivated reframe would have
   claimed the new target fixes everything; this one explicitly does not.

**Failures 1-2 were deflationary; failure 3 an omission of the framework's
own ground; failure 4 a target-alignment miss.** Together they show the
convergence/validation pattern carries a built-in bias toward shrinking
the claim, and is worst when the framework's distinctive constraints — or
the alignment between its signal and its GT — were never named first.

**The remedy (extended after failure 3):**

- **Read the cited paper carefully** — its architecture, layer, and
  decomposition axis — before applying the pattern. A title/abstract
  match is not a contribution match. (Catches failures 1 and 2.)
- **Before applying convergence reasoning at all, explicitly name the
  framework's distinctive constraints** — compute regime, deployment
  target, operational limits. Compare prior art *against that named
  ground*, not against a generic version of the framework. (Catches
  failure 3.)
- Ask of each prior work: same *layer* (model vs measurement)? same
  *category* (toolkit vs framework)? same *decomposition level* (causal
  source vs measurement signal)? same *compute regime*? If not on any
  axis, it is complementary, not displacing.
- **Before pre-registering a validation, anchor the GT on signal-target
  alignment, not data availability** (catches failure 4). Ask: *what does
  my signal semantically measure, and what GT is the most aligned target?*
  — then choose operationalization. Pre-registration ensures a prediction
  is tested; it does not ensure it is tested against the *right* target.
  Target-alignment review belongs *before* the pre-registration commits.
  (And: when the aligned target still can't be cleanly measured — as here,
  where diagnostic accuracy recurs the admission skew — say so; alignment
  doesn't imply tractability.)

**The discipline applies recursively to itself.** Calibrated-claims means
*accurate* claims — accurate distinctions between architecturally-
different prior art, accurate naming of the framework's own
differentiators, and no deflation below what the evidence supports. After
two deflationary misreads and one omission, the calibration check in
`contribution_shape_post_literature.md` flags risk in *both* directions:
under-claiming the empirical results and the constraint demonstration, and
over-claiming by treating the constraint as a result rather than a
demonstrated capability.

These calibration-failure modes were not anticipated by the original
convergence observation above; they are its first correction.

### The fourth-order pattern: positioning substituting for experiment

The three failures above share a deeper signal. The contribution claim
was re-derived three times in three turns (synthesis-not-novel →
measurement-layer + over-deflation → compute-constraint-first). Each
reframe was locally justified by a real observation; the *cumulative*
effect is contribution-claim instability — oscillation, not calibration.
The discipline pattern was meant to produce calibrated claims; it did not
catch this because the failure mode was orthogonal to what this document
tracked (it tracked literature-convergence misreads, not the meta-failure
of reframing-in-advance-of-evidence).

The remedy is not more elaborate convergence documentation — that would
be the same failure mode one level up. The remedy is to **recognize when
positioning work is substituting for experimental work**, and stop. The
contribution claim is adjudicated by **stage-6 chest-pain measurements**,
not by analysis. Further refinement of the claim in the abstract, ahead
of that data, is positioning proliferation.

**This document is FROZEN as of 2026-05-23.** No further entries until a
new calibration-failure pattern surfaces *during substantive work*
(experiment execution, not reframing). The next contribution-claim
revision is triggered by stage-6 data. See the freeze note in
`contribution_shape_post_literature.md`.

## Calibration failure #5 (2026-05-26): correctness-prediction conflated with uncertainty-measurement

**Unfreeze condition met.** This surfaced *during substantive work* —
building and running the semantic-entropy comparator (`11_...py`) and the
user scrutinizing what AUROC-against-correctness actually validates — not
during reframing. That is exactly the trigger the freeze reserved for.

**The failure.** Throughout the framework's empirical work, *correctness
prediction* and *uncertainty measurement* were treated as the same target.
They are not. Correctness is a property of the output vs ground truth.
Uncertainty is a property of the model's distribution over outputs.
Measuring AUROC of a signal against correctness tests whether the signal
**predicts correctness**; it does **not** test whether the signal
**measures uncertainty** as a latent construct. The cross-domain
`mean_entropy` results (AUC 0.66–0.69) are correctness-correlation results;
they demonstrate correctness-prediction, not uncertainty-measurement.

**Two precision points, opposite directions (both load-bearing):**

1. *Real narrowing.* The methods paper can claim correctness-prediction
   under deployment cost, validated on MCQ where correctness is
   unambiguous. It cannot, on this evidence, claim it measures uncertainty
   in a deeper epistemic sense. Establishing uncertainty-*measurement*
   would need convergent validity across distinct distributional measures,
   behavior under controlled uncertainty manipulation, and
   calibration-regime discrimination — none performed. (Phase-B's Spearman
   ~0.99 among `mean_entropy`/`mean_gap_top2`/`mean_p_max` is within-family
   convergent validity only; it does not extend to perturbation- or
   sampling-based measures.)

2. *Anti-over-deflation (the live risk per [[contribution_shape_post_literature]]).*
   The construct-validity gap is **the entire applied-UQ field's**, not a
   framework-specific lapse. Kuhn's semantic entropy, the Anatomy of
   Uncertainty paper, and the selective-prediction/deferral literature all
   validate by AUROC-against-correctness. So the honest statement is "the
   framework makes a correctness-prediction claim, validated as the field
   validates such claims" — NOT "the framework is uniquely sloppy about
   uncertainty." For the deployment use case (flag likely-wrong LLM
   outputs) correctness-correlation is precisely the relevant property.

**The fifth-order pattern (what #5 adds beyond #1–4).** The four prior
failures each conflated two architecturally-distinct things (toolkit vs
framework; model-layer vs measurement-layer; methodology-novelty vs
deployment-differentiation; target-alignment vs data-availability). #5 is
the same shape again (correctness-prediction vs uncertainty-measurement).
The through-line: **conceptual precision degrades under positioning
pressure**, and the discipline pattern *catches* failures (each via an
external flag from the PI) but does not *prevent* them. Refinement to the
discipline, recorded: positioning work that narrows a contribution claim
must also state **which specific claim is validated by which specific
empirical content, and where the conceptual edge is** — not just narrow the
headline. The methods-paper contribution narrative narrowed repeatedly
without naming, at each step, which distinction was being drawn.

**Remedy for the comparator (executed, not deferred):** the
`11_semantic_entropy_comparator.py` `_evaluate` and the pre-reg
(`prereg_semantic_entropy_comparator.md`) report (1) correctness-prediction
parity and (2) signal agreement as **separate** questions, and state
explicitly that neither establishes uncertainty-measurement. The MCQ
degeneracy (semantic entropy → letter-agreement entropy on a tiny support)
is registered in advance so signal agreement is not over-read as convergent
validity.

**Methods-paper consequence (flagged, NOT executed — drafting stays
deferred):** `docs/paper/draft.md` §1/§2 likely treat correctness-prediction
and uncertainty-measurement as equivalent and need a precision pass — make
every claim specify correctness-prediction vs uncertainty-measurement. This
pass waits until the comparator lands, per the drafting-deferral discipline.

## Positive instance: discipline pattern on a *favorable* result (2026-05-26)

The calibration-failure log above is all negative-valence (misreads,
omissions, walk-backs). The semantic-entropy comparator
(`2026-05-26-medqa-semantic-entropy-comparator.md`) is the first recorded
test of the discipline pattern on a result that *favors* the framework —
and favorable-result handling is the pattern's hardest test, because the
temptations invert: soften the caveats, inflate the scope, recover narrowed
claims. Worked example of doing it right:

- The result (single-run AUC 0.762 vs semantic 0.595, lift −0.167 CI
  excludes 0) is real and favorable to the compute-constraint claim.
- It was framed as "cheap and expensive measure *different* things; cheap
  predicts correctness better *here*" — NOT the easier "cheap approximates
  expensive" (the Spearman 0.31 forbids the latter).
- Five caveats kept explicit, not softened (correctness≠uncertainty,
  MCQ-structural advantage, 20% forced extraction, Anatomy-corroboration =
  known-weakness-surfacing, N=150 single-cohort).
- The cost claim was *corrected against the framework's favor's
  understatement* ("1/6" → ~2–3 orders of magnitude) — accuracy, not
  cheerleading.
- It was recorded as a **supportive data point, gated** behind a robustness
  check (full N + second model ± law) before it may anchor the methods-paper
  §1 — explicitly resisting the over-reach of anchoring a paper on one
  N=150 cohort. The prior contribution-claim narrowings were NOT recovered.

This strengthens item 5 of `contribution_shape_post_literature.md` (the
discipline-pattern contribution): it now has worked examples of *both*
negative-result handling (P5/P6 inversion, P4 walk-back) and positive-result
handling (this comparator), held to the same rigor. Integrity is handling
both valences identically.

## Cross-references

- `perturbation_uq_literature.md` (families 1–3 in detail).
- `feedback_calibrated_claims.md` (claim strength matched to evidence).
- `project_diagnose_rather_than_reframe.md` (the discipline applied to
  the caveat above).
- `project_measurement_protocol_as_contribution.md` (the protocol-framing
  this observation reinforces).
