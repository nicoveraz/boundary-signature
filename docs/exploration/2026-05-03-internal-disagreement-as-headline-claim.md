# Internal disagreement as the framework's headline claim

**Date:** 2026-05-03
**Status:** working synthesis, written for self before stage 3.3b's design pass

A working frame, not a polished document. The act of writing forces
the framing decision that propagates to methods paper, gate-experiment
analysis, and the design choices in stage 3.3b. Half an hour of writing
now beats unsystematic framing under deadline pressure later.

## What the framework was originally conceived as

Boundary-aware reasoning: detect when an LLM's reasoning trajectory
enters a region where its training distribution doesn't support
reliable inference. Originally framed via assembly theory and
structural signatures over recovered reasoning graphs. The intuition:
the framework would flag trajectories operating in "unfamiliar"
structural regions, regardless of whether the model itself signaled
uncertainty.

## What the literature does

LLM uncertainty estimation has converged on **confidence calibration**.
Three dominant approaches:

- **Prompt-elicited confidence** (Lin, Kadavath, Tian): ask the model
  to produce a probability or verbal confidence; evaluate whether that
  probability matches actual error rate.
- **Ensemble disagreement** (Singhal et al.'s Med-PaLM): multiple
  samples; evaluate whether they agree.
- **Logit inspection** (Kadavath, Hager): inspect probability mass at
  the answer position; evaluate calibration of those probabilities.

All three share an assumption: **the model's expressed or readable
confidence on its final output is the right object to study**.
Calibration improves when expressed confidence matches actual error
rate. Deferral decisions are made on the basis of expressed
confidence.

This framing leaves a class of failure modes uncovered.

## What F7 demonstrates

The Condition C end-to-end dress rehearsal on row 0 of MedQA test
split (`docs/exploration/condition_c_end_to_end_2026-05-03.md`)
surfaced a finding the confidence-calibration framing cannot
accommodate:

- The **CoT-final-answer prompt** produced the correct answer (B) with
  no hedging. The model is confident in its final answer.
- The **hypothesis-distribution-query prompt**, asked to estimate the
  probability of each answer at each reasoning step, **persistently
  favored a different answer (A)**. Across all five timesteps
  (prior + 4 reasoning steps), argmax stayed on A.

The model has two ways of evaluating the same reasoning, and they
disagree. The CoT path integrates step 4's chain-of-command nuance
and commits to B. The hypothesis-distribution path anchors on early
reasoning and never updates much.

A confidence-based deferral method would not flag this case. Condition
B asks "how confident are you in your final answer?" The model has
just produced B with no hedging; it would express high confidence.
Confidence-based deferral keeps the case. It happens to be correct,
so no immediate harm.

But the trajectory was *unstable*. Small variations in how the question
is elicited produce different answers. The visible outcome depended on
a specific prompt structure (CoT-with-final-answer rather than
distribution-at-each-step). The model's apparent confidence on its
final answer was deceptive — it concealed that the model was
internally divided about the answer.

## How structural signature could detect this

The framework's signature components are computed on the recovered
graph and the per-trajectory distributions. Two paths are plausible
for catching internal-disagreement cases like F7:

**Path 1: voi_flatness × entropy_plateau combination.** voi_flatness
measures mean information value of the trajectory's edges (computed
from the recovered graph: actions that historically separate diagnoses
have high VoI). entropy_plateau measures the slope of the trajectory's
own per-step distribution entropy. The signal of internal disagreement
is **high voi_flatness AND near-zero entropy_plateau slope**: the
actions taken were informative ones (in the recovered graph), but this
trajectory's distributions didn't reflect the information. The model
took information-rich actions without updating its expressed beliefs.

**Path 2: cross-trajectory comparison at later timesteps.** If most
B-final-answer trajectories at step 4 have argmax-shifting-toward-B
by that point, this trajectory's argmax-still-on-A pattern is
distributionally unusual. distance-from-trajectory at the step-4
state-embedding would be high if the embedding model picks up on the
distribution-shape difference. Whether it does is empirical and only
testable at stage 4 with real e5-large embeddings on real MedQA
trajectories.

Neither path is yet validated. F7 is a single observation; the
framework's premise is that signature components catch this class of
failure at scale. Stage 4 either confirms or refutes that.

What is clear: confidence-based methods structurally cannot see this
failure mode. They evaluate the wrong object (final-answer
confidence, not internal trajectory consistency).

## The reframed claim

The framework's novelty is **not** "we detect when models are wrong."
Confidence-based methods do that adequately for the cases where the
model knows it's uncertain.

The framework's novelty is:

> **We detect a specific failure mode — internal trajectory
> disagreement — that surface methods (confidence calibration,
> ensemble disagreement, logit inspection) structurally cannot see.**

This positioning is more defensible than "better at detecting wrong
answers" for three reasons:

1. **It's true.** F7 is a worked example where confidence calibration
   genuinely cannot detect the failure mode the framework targets.
   The mechanism is structural: confidence is on the final answer,
   internal disagreement is between trajectory and final answer.
2. **It's novel.** No existing method explicitly targets internal
   trajectory disagreement. Singhal-style ensemble disagreement is
   the closest analog but it's about disagreement across samples
   from the model, not within a single sample's reasoning trajectory.
3. **It's clinically meaningful.** A model that confidently produces
   correct answers via unstable trajectories is a model whose errors
   surprise clinicians. Internal trajectory disagreement is the kind
   of signal clinicians naturally want flagged: "this case looks
   confident but the model's reasoning could have gone the other
   way."

## Implications for stage 3.3b design

### action_id design: text-only is correct *because* of the reframing

The reframing initially seemed to suggest ``ActionCanonicalizer``
should take per-step distributions as input — if the framework's
signal is about distribution shifts, action_ids should encode those
shifts. On careful examination, **this is wrong**. Text-only
``ActionCanonicalizer`` (the original hybrid design from stage 3.2's
deferred C1 question) is correct precisely *because* of the reframing.

Working through the three options:

- **(a) Text-only** (or hybrid: text + position + embedding-bin).
  Canonicalize over reasoning-step content. Two trajectories with
  content-similar reasoning get the same action_id regardless of how
  their per-step distributions shifted.
- **(b) Shift-aware.** Canonicalize over reasoning step + source
  distribution + target distribution. Two trajectories with similar
  content but different distribution shifts get *different*
  action_ids.

For an F7-shaped case under (a) text-only: the trajectory's action at
"step 4" (chain-of-command consideration) gets an action_id. Other
trajectories with similar chain-of-command reasoning get the **same**
action_id. Recovery aggregates: edges with this action_id have
varied targets — some trajectories ended at A (as in F7), some at B.
**That variation IS the signal.** Edge-level VoI on this action is
high (information value of the action varies across trajectories).
The framework's claim — "internal trajectory disagreement is
detectable" — manifests as **anomalous edge attributes in the
recovered graph**, not as anomalous action canonicalization.

Under (b) shift-aware: the F7 trajectory's action produces a unique
action_id (because its source-target distribution-shift is
distinctive). Other trajectories with similar content but different
shifts get different action_ids. Recovery aggregates *less* — edges
with the same action_id all have similar shifts, so edge-level VoI is
*low*. **The signal is canonicalized away.**

So: **text-only ActionCanonicalizer is correct because the framework's
signal lives at the graph-edge level, not the action-identity level.**
The recovered graph's edge-attribute computation (recovery.py's VoI
calculation) is what surfaces internal disagreement; action
canonicalization just needs to aggregate "the same action" across
trajectories so edges can accumulate meaningful statistics.

The temptation to make ``ActionCanonicalizer`` "shift-aware" because
the reframing emphasizes distribution shifts is a category error —
canonicalization is about identity, not behavior; behavior lives on
the recovered graph's edges.

**Stage 3.3b decision locked: text-only-with-embedding-bin
ActionCanonicalizer (the original hybrid plan from stage 3.2's
deferred C1 question). The reframing reinforces rather than
complicates this design.** Document this reasoning in stage 3.3b's
implementation so future-readers don't reopen the question.

### Hypothesis-distribution batching across questions

F8 (model anchors) suggests per-step distributions are highly
correlated within a question. The interesting signal is in
cross-trajectory comparison at the same timestep — exactly what
Path 2 above relies on. Batching that produces aligned per-timestep
slices across questions is operationally valuable for the recovered
graph's ability to support cross-trajectory analysis at each
timestep. Worth surfacing in 3.3b's design pass.

### Gate metric (deferred)

The reframing implies the meaningful gate is "Condition C beats
Condition B on cases where Condition B is uninformative" rather than
"Condition C deferral-curve AUC ≥ 0.65 on the full dataset." Not
blocking stage 3.3b. Worth recording as ADR-0006 once stage 4 setup
begins; the dual-gate option (original AUC for funding-deliverable,
subset AUC for paper headline) is the leading candidate.

### Peer-review defenses (deferred)

The reframed claim will face challenges from at least four literatures:
prompt-elicited / logit-inspection confidence (Kadavath, Tian),
ensemble disagreement (Singhal et al.), self-consistency / majority
voting (Wang et al.), and verifier-model approaches (Saunders et al.
+ process-supervision literature). Each has a clean defense; the
defenses are sharper if articulated now rather than during paper
revision. Half a page added to this document during the next
docs-only session.

## Methods-paper consequences

When the chest-pain gate experiment writes up, the framing is:

> We propose structural-signature monitoring as a **complement** to
> confidence-based deferral, designed to detect a failure mode
> (internal trajectory disagreement) that confidence-based methods
> structurally cannot detect. We demonstrate the failure mode is
> real and present in qwen2.5:7b on MedQA-USMLE [F7 worked example,
> docs/exploration/condition_c_end_to_end_2026-05-03.md]. We
> characterize the framework's detection of this failure mode via
> [stage 4 results].

Not:

> We propose structural-signature monitoring as a **better** way to
> detect when LLMs are wrong.

The first framing is defensible and novel. The second framing is
overclaiming and would invite immediate "but X confidence method also
detects this" comparisons. The first framing is also more honest:
the framework targets a specific class of failures, not the entirety
of "wrong-answer detection." Confidence calibration retains its
value for the cases it covers; the framework adds coverage of cases
it doesn't.

This positioning also clarifies what stage 4's headline metric needs
to do: not just "AUC ≥ 0.65 on the deferral curve" but specifically
"AUC ≥ 0.65 on cases where confidence-based methods cannot
distinguish correct from incorrect." That's a different — and more
meaningful — gate than the current spec. Worth revisiting the gate
criteria during stage 4 setup.
