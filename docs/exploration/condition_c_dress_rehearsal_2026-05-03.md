# Condition C dress rehearsal (qwen2.5:7b on three MedQA questions)

**Date:** 2026-05-03
**Source:** `GBaker/MedQA-USMLE-4-options`, test split, rows 0 / 100 / 500
**Model:** `qwen2.5:7b-instruct` via Ollama
**Generation parameters:** `temperature=0.0`, `seed=42`
**Decomposer:** stage-3.2 default (case-insensitive regex, graceful, min=3 max=10)

Manual walk-through of what Condition C will do, run against three
real MedQA questions, before stage 3.3's design pass starts. Captures
both the format / parsing behavior AND a striking calibration finding
that affects the design pass's expectations.

## Rehearsal protocol

For each of test rows 0, 100, 500:

1. **Initial CoT generation.** Same structured-format prompt as the
   stage 3.1 grounded exploration ("Reasoning step N: ... Final
   answer: X").
2. **Decompose** the LLM output via the new `Decomposer`.
3. **Per-step hypothesis-distribution query** at the middle reasoning
   step. Prompt asks for a JSON object with keys A/B/C/D and float
   values summing to 1.0.
4. **Parse and analyze** the distribution: validity, sum-to-1
   normalization, calibration (peaked vs uniform), agreement with
   the eventual final answer.

## Summary table

| Row | meta_info | Q chars | Steps | Predicted | Correct | Result |
|---|---|---|---|---|---|---|
| 0 | step1 | 608 | 3 | A | B | wrong |
| 100 | step2&3 | 890 | 3 | B | D | wrong |
| 500 | step1 | 648 | 4 | D | B | wrong |

**3/3 wrong.** This is small-N and the temperature-0 sampling makes
each run deterministic, but the consistency is striking. qwen2.5:7b is
either materially weaker at MedQA than the published 4-option
benchmarks suggest, or this 3-question sample was unlucky, or the
temp=0 path systematically produces a particular kind of wrong
reasoning that's prevalent in the model. Worth a wider sweep at H100
time to characterize.

## Hypothesis-distribution outputs (verbatim)

### Row 0 (after reasoning step 2 of 3)
```
{"A": 0.6, "B": 0.2, "C": 0.05, "D": 0.15}
```
Peak: A (0.6). Correct: B. **Confidently wrong.**

### Row 100 (after reasoning step 2 of 3)
```
{"A": 0.05, "B": 0.8, "C": 0.05, "D": 0.1}
```
Peak: B (0.8). Correct: D. **Confidently wrong.**

### Row 500 (after reasoning step 3 of 4)
```
{"A": 0.05, "B": 0.05, "C": 0.1, "D": 0.8}
```
Peak: D (0.8). Correct: B. **Confidently wrong.**

## Findings that inform stage 3.3 design

### F1: JSON parsing is trivial at temp=0

All three hypothesis-distribution queries returned valid JSON in 42
chars exactly, no markdown fences, no prose preamble, no trailing
commentary. `json.loads()` parses each directly.

**Implication for Condition C:** the canonical happy-path JSON parser
suffices for temp=0 instruction-tuned models. The repair-prompt
fallback path (re-issue with `prompts/repair.txt` if parsing fails)
remains important for higher-temperature runs and for less compliant
models, but the temp=0 chest-pain gate run can probably skip the
repair path 99%+ of the time.

### F2: Probabilities sum to 1.0 exactly

In every sample. The model has been RLHF'd to produce normalized
distributions when asked. **No softmax-renormalization needed in the
parser.**

### F3: Distributions are peaked, not uniform

Peak values: 0.6, 0.8, 0.8. Mean entropy across the three:
H(0.6, 0.2, 0.05, 0.15) ≈ 1.45 bits
H(0.05, 0.8, 0.05, 0.1) ≈ 0.96 bits
H(0.05, 0.05, 0.1, 0.8) ≈ 1.02 bits

Mean: ~1.14 bits out of max 2 bits (uniform = 2 bits for 4 choices).
The model commits — it doesn't hedge with near-uniform distributions.

### F4: Hypothesis distribution agrees with the final answer in every case

For all three rows, `argmax(hypothesis_distribution) ==
final_answer_letter`. This is at the **middle** reasoning step (step
2 of 3 for rows 0 and 100; step 3 of 4 for row 500). The model has
already committed by mid-trajectory.

**Implication for `entropy_plateau`:** at temp=0, the model commits
early and stays committed. The slope of entropy across timesteps may
be consistently small (close-to-zero plateau) for many trajectories —
which is fine because that's what `entropy_plateau` is named for, but
worth knowing it's the dominant pattern rather than a rare boundary
case at this model+temperature.

For temperature > 0 or for less committal models, expect more varied
slopes. The chest-pain gate experiment will need to characterize the
distribution of `entropy_plateau` values across its dataset, not just
look for outliers.

### F5: Confidence is ~0.6-0.8 on wrong answers

This is **the failure mode that motivates the project**. Confidence-
based deferral (Condition B) would not flag any of these three cases
— the model signals 60-80% confidence in its (wrong) answer. The
threshold for "low-confidence -> defer" would have to be set above
0.8 to catch the row-0 case (0.6 confidence on wrong A), and above
0.8 still wouldn't catch rows 100 and 500.

**The framework's bet:** structural signature components catch these
boundary cases via signals other than the LLM's expressed confidence
— specifically, distance-from-trajectory (these reasoning patterns
should be unusual in the recovered graph) and voi_flatness (the
specific actions taken at each step should land in low-VoI / high-
ritualization regions).

### F6: Per-step distribution agrees with full-CoT distribution

A subtle but important methodological point: the partial-reasoning
hypothesis distribution at step k commits to the same answer as the
full-CoT final answer. This means the per-step distributions are
NOT independent samples of the model's "evolving belief"; they're
strongly correlated with the model's mid-trajectory commitment.

For `entropy_plateau` to detect "model is gaining uncertainty"
behavior, the trajectory needs to either (a) have steps that produce
genuinely different distributions (not what we observe at temp=0),
or (b) produce distributions that vary in ways the slope captures
even if the argmax is stable.

The latter is where the signal might live: even if argmax stays at
A, the probability mass on A might shift from 0.4 → 0.5 → 0.6 → 0.7
across steps (entropy decreasing, model converging) versus 0.6 → 0.55
→ 0.5 (entropy increasing, model gaining doubt). The slope of
entropy detects the latter as a boundary signal.

This rehearsal didn't run all per-step distributions to characterize
the across-step variance — that would require ~3-4 hypothesis
queries per question. Worth adding that to a future exploration if
the entropy_plateau component looks weak in stage 4 results.

## Hypothesis-distribution prompt template (working version)

This prompt produced clean JSON in 3/3 trials at temp=0:

```
Given the following USMLE question and partial reasoning, estimate
the probability that each answer choice is correct.

Question:
{question}

Choices:
{letter}: {text}
... (one per line)

Reasoning so far:
Reasoning step 1: {step_1_text}
Reasoning step 2: {step_2_text}
... (up to step k)

Output your probability estimates as a JSON object with keys "A",
"B", "C", "D" and float values between 0 and 1 that sum to 1.0.
Output ONLY the JSON, nothing else.

JSON:
```

This becomes the basis for `prompts/condition_c_hypothesis.txt` in
stage 3.3. Three small refinements worth considering during the
design pass:

1. **Make the keys configurable** to handle MMLU 4-letter or any
   future variant. Template substitution: `keys "A", "B", "C", "D"`
   becomes `keys {", ".join(f'"{k}"' for k in hypothesis_space)}`.
2. **Drop the "Output ONLY the JSON, nothing else" instruction** for
   models that ignore it. The parser strips markdown fences anyway
   per the dress-rehearsal cleanup helper. The instruction is helpful
   when honored, harmless when ignored.
3. **Consider asking for log-probabilities or odds** rather than
   probabilities. RLHF'd models may anchor on round-number
   probabilities (0.6, 0.8) more than on log-odds. Empirical question
   for stage 4.

## Reproducibility

The script that produced this rehearsal is preserved in this
session's transcript. To reproduce:

```python
from datasets import load_dataset
import httpx, json
from bsig.medqa import Decomposer

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
decomposer = Decomposer()
client = httpx.Client(timeout=180.0)

def call(prompt):
    resp = client.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:7b-instruct", "prompt": prompt,
              "stream": False, "options": {"temperature": 0.0, "seed": 42}},
    )
    return resp.json()["response"]

# (CoT prompt + decomposer.decompose + hypothesis prompt + json.loads
# per the rehearsal protocol above)
```

## Implications worth surfacing for stage 3.3 design pass

1. **The hypothesis-distribution prompt design is largely settled.**
   The temp=0 happy path works cleanly; the design pass mostly needs
   to make the keys configurable and decide how to handle the
   higher-temperature pathologies that don't show up here.

2. **Repair-prompt loop importance is reduced for temp=0.** Still
   needed for robustness, but the chest-pain gate experiment can
   probably set retry_count low (1-2) and rely on the canonical
   parser at temp=0.

3. **Calibration is overconfident.** Confidence 0.6-0.8 on wrong
   answers means the framework's value is real and not just
   theoretical — the failure mode the structural signature targets
   is the dominant failure mode at this model+benchmark combination.

4. **`entropy_plateau` slope characterization needs empirical work.**
   At temp=0 the model commits early; the slope distribution is
   probably narrow. Stage 4's analysis must include a slope-
   distribution histogram before drawing conclusions about whether
   the component carries signal.

5. **The 3/3 wrong rate is striking and worth a sanity-check sweep.**
   If qwen2.5:7b really gets MedQA at 30% accuracy or below, that
   significantly changes the gate experiment's expectations. Run
   accuracy on the full 1273 test set as part of stage 4's setup;
   if accuracy is much lower than published benchmarks (Singhal et al.
   reported Med-PaLM at 67% on MedQA-USMLE), investigate whether
   we're using the model wrong (prompt format, temperature, model
   variant, quantization) before drawing conclusions.

## Save state

This file + the qwen2.5 failure-case file at
`docs/exploration/qwen25_failure_case_2026-05-03.md` together provide
the empirical anchor for stage 4's validation:
- Failure case: known boundary case, signature should flag.
- Dress rehearsal: format / parsing / calibration patterns; three
  more boundary cases for cross-checking.

When stage 4 runs, locate questions by `medqa-test-{0,100,500}` in
the recovered graph; pull their signature scores; verify whether the
framework flagged all four (failure-case row 0 from prior exploration
+ rows 0/100/500 from this rehearsal — note row 0 is shared).
