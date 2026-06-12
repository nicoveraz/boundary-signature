# Condition C end-to-end dress rehearsal

**Date:** 2026-05-03
**Source:** `GBaker/MedQA-USMLE-4-options`, test split, **row 0** (same as failure-case exemplar)
**Model:** `qwen2.5:7b-instruct` via Ollama
**Generation parameters:** `temperature=0.0`, `seed=42`
**Prompts:** `condition_c_initial.txt` v1 (just written, this commit), inline hypothesis-distribution prompt
**Decomposer:** stage-3.2 default

Full Condition C flow on a single question:
1. Initial CoT via `condition_c_initial.txt` v1.
2. Decompose.
3. **For every timestep** (prior + after each reasoning step), query the
   hypothesis distribution.
4. Trace canonicalizer node_ids across timesteps.

Three findings that materially affect stage 3.3b's design pass and
stage 4's methodology.

## What happened (verbatim)

### Initial CoT output
```
Reasoning step 1: The resident has made a surgical error by cutting the
flexor tendon during carpal tunnel repair. This is a significant
complication that could impact the patient's recovery and function.

Reasoning step 2: According to medical ethics and standards, it is
essential to disclose all complications to patients, regardless of
their severity or potential impact on outcomes. Withholding information
can lead to mistrust and legal issues.

Reasoning step 3: The attending physician's suggestion to omit the
complication from the operative report and not inform the patient goes
against ethical guidelines and best practices in medical care.

Reasoning step 4: Reporting a colleague for unethical behavior, such
as failing to disclose a significant surgical error, is an extreme
measure that should be considered only if there are no other options.
The resident should first address the issue with the attending
physician directly.

Final answer: B
```

**The model got it right.** Correct answer is B; final answer is B. ✓

### Per-step hypothesis distributions
| Timestep | Distribution | argmax | Peak | Entropy |
|---|---|---|---|---|
| 0 (prior) | {A: 0.4, B: 0.3, C: 0.1, D: 0.2} | A | 0.40 | 1.846 bits |
| 1 (after step 1) | {A: 0.6, B: 0.2, C: 0.05, D: 0.15} | A | 0.60 | 1.533 bits |
| 2 (after step 2) | {A: 0.6, B: 0.2, C: 0.05, D: 0.15} | A | 0.60 | 1.533 bits |
| 3 (after step 3) | {A: 0.65, B: 0.2, C: 0.05, D: 0.1} | A | 0.65 | 1.417 bits |
| 4 (after step 4) | {A: 0.6, B: 0.25, C: 0.05, D: 0.1} | A | 0.60 | 1.490 bits |

**Argmax: A → A → A → A → A.** The hypothesis-distribution-query LLM
**never commits to B**, even after seeing reasoning step 4 (which is
the chain-of-command argument that drives the final answer to B).

**Entropy slope (entropy_plateau): -0.083 bits/step.** Decreasing —
the model converges. But it converges to the *wrong* answer
distribution-wise, even though the final answer is right.

## F7 (new): CoT-final-answer and hypothesis-distribution-query disagree

This is the most consequential finding. The same partial reasoning
gets two different verdicts depending on prompt:

- **CoT prompt** ("show reasoning, give final answer"): final answer
  B, correct.
- **Hypothesis-distribution prompt** ("estimate probability of each
  answer given reasoning so far"): argmax A, wrong.

The two prompts are evaluating the same input (4 reasoning steps)
through different lenses. The CoT prompt's "final answer" question
elicits a synthesis step that integrates all reasoning including the
chain-of-command consideration. The hypothesis-distribution prompt
treats each step's reasoning as evidence for a probability estimate
and the model anchors on early-step content.

**This is exactly the boundary case the framework targets.** The
trajectory has a striking internal disagreement between its
distributions (favor A) and its final answer (B). Confidence-based
deferral (Condition B) would not flag this — Condition B's
confidence is on the FINAL answer, which is correct here. The
structural signature should detect the disagreement via
distance-from-trajectory (this trajectory's distribution-pattern is
unusual — most correct-answer trajectories have argmax stable on the
correct letter throughout) or via voi_flatness (the action of
"reasoning step 4" had high VoI — it changed the final answer
without changing the per-step distribution).

This finding **strengthens the case for Condition C over Condition
B**: the structural-signature monitoring approach has a path to
detecting cases that confidence scoring genuinely cannot see.

## F8 (new): Model anchors on initial assessment, barely updates

Look at the per-step distributions: timestep 1 = timestep 2 EXACTLY.
{A: 0.6, B: 0.2, C: 0.05, D: 0.15}. The model assigns identical
probabilities after seeing step 2 as it did after seeing step 1.
Step 2 produced zero update.

Step 3 nudged: {A: 0.65, ...}. Step 4 nudged back: {A: 0.6, B: 0.25,
...}. The mass shift toward B is real (0.20 → 0.25) but small.

The hypothesis-distribution-query LLM is effectively **ignoring most
of the additional reasoning**. This is a separate failure mode from
F7 (the prompt-template disagreement). It's a property of how
instruction-tuned models respond to "evaluate this reasoning"
prompts: they anchor.

**Implication for `entropy_plateau`:** the metric will detect the
plateau pattern (entropy stops dropping after timestep 1), but the
plateau alone doesn't distinguish "model is correct and committed"
from "model anchored on wrong answer." `voi_flatness` and
`distance_from_trajectory` carry the discriminating signal.

## F9 (new): Prompt design substantially affects answer correctness

The `condition_c_initial.txt` v1 prompt (just written) differs from
the previous dress-rehearsal prompt in two small ways:
1. Adds: "Each reasoning step should be a distinct logical move
   (consider symptoms, identify the pattern, apply diagnostic
   criteria, eliminate alternatives, etc.)"
2. Changes "minimum 3, maximum 8" to "aim for 3-5 steps total"

Same model, same temperature, same seed, same question. Different
prompt → **different final answer**. The previous prompt produced 3
steps and answer A; the v1 prompt produces 4 steps and answer B.

The added "consider symptoms / identify pattern / apply criteria /
**eliminate alternatives**" framing biases the model toward more
careful reasoning that catches the chain-of-command issue. The
"eliminate alternatives" cue specifically maps to step 4's content
("Reporting a colleague... is an extreme measure...").

**Implication for stage 4 methodology:** the prompt is part of the
framework configuration and gets recorded in
`signature_metadata.prompt_versions` (per stage 3.1). But if
different prompts produce 50% vs 67% accuracy on the same model, the
framework's headline AUC depends on prompt choice as much as on the
signature components themselves. Stage 4's analysis must (a) report
prompt versions transparently, (b) ideally test multiple prompts to
characterize sensitivity, (c) treat the prompt as a hyperparameter
not a fixed input.

This also revises the failure-case exemplar's status. The previous
file (`qwen25_failure_case_2026-05-03.md`) anchored on row 0 as the
canonical "qwen2.5 fails this question" case. With the v1 prompts,
it's now: "qwen2.5 succeeds on this question with v1 prompts; fails
with the dress-rehearsal-prompt variant." That's a different
finding — prompt sensitivity rather than knowledge gap.

## F10: Prior distribution carries information

Timestep 0 (no reasoning, asked to estimate distribution given just
the question): {A: 0.4, B: 0.3, C: 0.1, D: 0.2}. **Argmax A — the
model's gut reads as "disclose unilaterally."** The post-reasoning
distributions push HARDER on A (0.6) before drifting back slightly.

The prior is meaningful and worth keeping (per design pass C2).
Without it, `entropy_plateau`'s slope would be calculated over
timesteps 1-4 only, which all have entropies between 1.42 and 1.53.
Slope would be near zero — no signal. Including the prior gives a
slope of -0.083, which is at least a measurable trajectory direction.

**Recommendation reaffirmed:** Condition C constructs the initial
state at timestep=0 with an LLM-queried prior distribution, even
though it costs one extra LLM call per question.

## F11: Canonicalizer node_ids vary per timestep as expected

```
t=0: node_id=9039bdbf3274449e... (steps=0)
t=1: node_id=472eb64185903c19... (steps=1)
t=2: node_id=a2e76599184fb2f3... (steps=2)
t=3: node_id=629395bf2b53a174... (steps=3)
t=4: node_id=76177073fb1c54a9... (steps=4)
```

Each timestep has a distinct node_id because the reasoning_steps
tuple changes. The mock embedder produces deterministic embeddings;
real embeddings (e5-large) would produce semantically-clustered
node_ids when reasoning text is similar across questions — that's
the recovery-aggregation use case.

## Implications for stage 3.3b design pass

1. **F7 (CoT vs hypothesis-distribution disagreement) is a load-
   bearing finding.** Condition C's value is exactly in capturing
   this disagreement via signature components. The design pass for
   3.3b should ensure the orchestration preserves both the
   distribution trajectory AND the final answer (and they may
   disagree — that's the signal).

2. **F8 (model anchors, barely updates) calibrates expectations for
   `entropy_plateau`.** The component will see decreasing-then-flat
   entropy patterns dominantly. Stage 4's analysis must histogram
   slope distributions rather than look for individual outliers.

3. **F9 (prompt sensitivity) is a methodological flag for stage 4,
   not a stage 3.3b implementation concern.** Worth recording the
   sensitivity finding so stage 4's experimental design accounts
   for it (multiple prompt variants ideally, or at minimum a
   single locked prompt with the sensitivity flagged).

4. **F10 (prior is meaningful) reaffirms design decision C2.**
   Initial state at timestep=0 with LLM-queried prior. Costs one
   extra LLM call per question; worth it.

5. **The Condition C orchestration spec is largely settled.** Initial
   CoT (1 call) → Decompose → Per-step distribution queries
   (N batched calls or N individual calls) → optional repair (≤1 per
   failure point) → Trajectory construction. Total LLM calls per
   question: 1 (initial CoT) + 1 (prior at timestep 0) + N (per-
   step distributions) + repairs. At N=4 for this rehearsal: 6 calls
   minimum, more if repairs fire. Repairs didn't fire here.

6. **JSON parsing was 100% reliable across 6 calls.** Repair-prompt
   path remains unused. The temp=0 happy path is genuinely happy.

## Reproducibility

Inline script in this session's transcript. Key commands:

```python
from datasets import load_dataset
from bsig.medqa import Decomposer, MCQStateCanonicalizer, load_prompt
import httpx, json

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
ex = ds[0]

initial_prompt = load_prompt("condition_c_initial").format(
    question=ex["question"],
    choices="\n".join(f"{k}. {v}" for k, v in ex["options"].items()),
)
# call Ollama with initial_prompt, decompose, then per-step distribution queries.
```

Re-running on the same Ollama install with the same prompt v1 should
reproduce. Bumping `condition_c_initial.txt` to v2 (any prompt revision)
will likely change results; record the version in
`signature_metadata.prompt_versions` per stage 3.1.

## Save state

Three exploration files now anchor the framework's empirical
characterization:

- `qwen25_failure_case_2026-05-03.md` — row 0, dress-rehearsal-
  variant prompt, qwen2.5 confidently wrong on A. **Status: revise
  to note that v1 prompts produce a different result.**
- `condition_c_dress_rehearsal_2026-05-03.md` — three rows
  (0, 100, 500), middle-step hypothesis distributions only, all
  three wrong with 0.6-0.8 confidence on wrong answer.
- This file — full Condition C flow on row 0, all 5 timesteps'
  distributions, surfaces F7/F8/F9 findings.

Together they characterize: prompt sensitivity, hypothesis-vs-CoT
disagreement, model-anchoring behavior. The empirical foundation is
substantially richer than it was at the start of stage 3.
