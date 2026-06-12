# Worked example: qwen2.5:7b confidently wrong on MedQA test row 0

**Date:** 2026-05-03
**Source:** `GBaker/MedQA-USMLE-4-options`, test split, row 0
**Model:** `qwen2.5:7b-instruct` via Ollama
**Generation parameters:** `temperature=0.0`, `seed=42`

This is the first piece of empirical evidence that the boundary-signature
framework's premise is operationally tractable: a real LLM produced
plausible-sounding reasoning leading to a wrong answer on a real
benchmark question. Saved here as a future validation anchor — when
stage 4's H100 run computes structural signature scores against this
exact question, we will know whether the signature actually detected
the failure mode.

A single observation is not validation. But the failure mode the
project's premise predicts (confident, plausible-sounding,
clinically-nuanced wrong reasoning) is present in qwen2.5:7b at this
scale. The signature has something to detect.

## The question (verbatim, USMLE step1)

> A junior orthopaedic surgery resident is completing a carpal tunnel
> repair with the department chairman as the attending physician.
> During the case, the resident inadvertently cuts a flexor tendon.
> The tendon is repaired without complication. The attending tells the
> resident that the patient will do fine, and there is no need to
> report this minor complication that will not harm the patient, as he
> does not want to make the patient worry unnecessarily. He tells the
> resident to leave this complication out of the operative report.
> Which of the following is the correct next action for the resident
> to take?

## Choices

- **A.** Disclose the error to the patient and put it in the operative report
- **B.** Tell the attending that he cannot fail to disclose this mistake ← correct
- **C.** Report the physician to the ethics committee
- **D.** Refuse to dictate the operative report

## qwen2.5:7b output (structured-CoT prompt)

```
Reasoning step 1: The resident is required to maintain accurate and
complete medical records for the patient's care. This includes
documenting all surgical complications, even minor ones, as they are
part of the patient's medical history.

Reasoning step 2: Ethical standards in medicine mandate transparency
with patients regarding their health information. Withholding this
information could be considered unethical and potentially harmful to
the patient if the complication were to recur or affect future care.

Reasoning step 3: The attending physician's suggestion to omit the
complication from the operative report is not aligned with medical
ethics, legal requirements, and best practices in patient care.

Final answer: A
```

## Why this is a boundary case

The clinically-correct action is **B** — escalate the disagreement
with the attending before any unilateral disclosure. The chain-of-
command nuance (you do not bypass the attending; you confront them
first) is clinically routine but not obviously connected to the
ethical principles the LLM reasoned over.

The LLM's reasoning is **plausible**. Each step is independently
defensible: documentation completeness, patient transparency,
attending's instruction misaligned with ethics. The argument leads
directly to "disclose to the patient" (option A) by composition of
those principles.

The reasoning is **wrong** because it skips a clinical-ethics step
that's not explicitly invoked: the chain-of-command obligation that
makes confronting the attending the correct first action. The LLM did
not reach for that step, and confidence-based deferral (Condition B)
would not flag this trajectory — qwen2.5 produced this output with
no hedging, no expressed uncertainty.

## What the framework predicts

If the project's bet is right, the structural signature components
should detect this trajectory as boundary-case:

- **`distance_from_trajectory`**: the reasoning trajectory occupies a
  region of the assembly graph where peer trajectories at similar
  timesteps did not converge to A. Specifically, the third reasoning
  step ("attending's suggestion is not aligned with ethics") sits in
  a region where most trajectories from comparable starting states
  reached B-aligned conclusions, not A-aligned ones. **High distance.**
- **`voi_flatness`**: the per-step actions (drawing on documentation,
  transparency, ethics-vs-instruction) have moderate VoI individually
  but the trajectory's transition pattern at the chain-of-command
  step is rare in the recovered graph. Specifically, the absence of
  a "consider chain of command" reasoning step is itself low-VoI by
  omission. Hard to predict the exact direction without running it.
- **`entropy_plateau`**: the hypothesis distribution converged
  early on A — qwen2.5's reasoning is monotonically supportive of A
  with no consideration of alternatives. **Strongly negative slope
  → high "plateau" signal in the rank-percentile direction the
  framework treats as boundary.**

## Operational use

When stage 4 runs:

1. Locate this question in the recovered MedQA graph by `question_id =
   "medqa-test-0"` (per stage 3.1's loader-synthesis convention).
2. Pull the trajectory's signature scores from the run's
   `signature_scores.parquet`.
3. Verify: was the composite score above the median for test
   trajectories? Above the 75th percentile? In the top 10%?

If yes — the framework detected this case. Useful methods-paper
anchor: "Here is a confidently-wrong LLM trajectory; here is the
structural signature score; here is why the framework flagged it
even though confidence-based deferral did not."

If no — the framework missed this case. Worth investigating which
component dragged the score down (probably distance, given the
small recovered-graph density at H100 scale) and whether tuning
or component-reweighting would catch similar cases.

Either way, the worked example anchors a concrete validation
moment.

## Update (same date, different finding): prompt sensitivity

The end-to-end Condition C dress rehearsal at
`condition_c_end_to_end_2026-05-03.md` re-ran row 0 with the just-
written `condition_c_initial.txt` v1 prompt (which adds "Each
reasoning step should be a distinct logical move" framing). With the
v1 prompt, qwen2.5:7b at temp=0/seed=42 produced 4 reasoning steps
and **the correct answer B**.

So this exemplar is now: "qwen2.5:7b confidently fails this question
with the dress-rehearsal-variant prompt, succeeds with the v1
prompt." That's prompt sensitivity rather than a knowledge gap.

The case is still useful as a worked example for stage-4 validation —
particularly because the per-step hypothesis distributions under the
v1 prompt STILL favor A (the wrong answer) even though the final
answer is B (correct). The CoT-vs-hypothesis-distribution
disagreement is exactly the failure mode the framework targets, and
it persists even when the final answer is correct. See finding F7
in the end-to-end dress rehearsal file.

## Reproducibility

```python
from datasets import load_dataset
import httpx

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
ex = ds[0]
prompt = f"""You are answering a USMLE-style clinical question. You MUST show your reasoning before giving the answer.

Question:
{ex["question"]}

Answer choices:
{chr(10).join(f"{k}. {v}" for k, v in ex["options"].items())}

Format your response EXACTLY like this:
Reasoning step 1: <your first reasoning step>
Reasoning step 2: <your second reasoning step>
Reasoning step 3: <your third reasoning step>
(continue with as many steps as needed, minimum 3, maximum 8)

Final answer: <single letter A, B, C, or D>
"""
resp = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:7b-instruct",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    },
    timeout=180.0,
)
print(resp.json()["response"])
```

Re-run on the same Ollama install reproduces the output above.
Different model versions or quantizations may produce different
output; this snapshot reflects qwen2.5:7b-instruct as of 2026-05-03.
