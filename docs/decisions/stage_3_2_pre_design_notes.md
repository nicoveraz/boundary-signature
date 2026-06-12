# Stage 3.2 pre-design notes

Grounded exploration findings from 2026-05-03, captured before the
stage 3.2 design pass. Inputs to the Decomposer / loader / trajectory-
construction decisions.

## MedQA dataset shape (`GBaker/MedQA-USMLE-4-options`)

- 10,178 train + 1,273 test. Test count matches CLAUDE.md §7's quote.
- Always 4 options `{A, B, C, D}`. See `ADR-0003` for variant decision.
- Schema: `question` (median 715 chars, max 3543), `options` (dict),
  `answer` (full text), `answer_idx` (correct letter), `meta_info`
  (`step1` / `step2&3`), `metamap_phrases` (list, framework discards).
- Class balance roughly even across A/B/C/D. `meta_info` ~53/47.
- 0/1273 mismatches between `answer` and `options[answer_idx]`. Clean.
- No native question_id; loader synthesizes `medqa-{split}-{idx}`,
  stable across loads of the same dataset version.

## LLM CoT output shape (qwen2.5:7b-instruct via Ollama, temp=0, seed=42)

### Vanilla "think step-by-step" prompt
Model went straight to a one-letter answer with NO reasoning. Modern
instruction-tuned models ignore vague "think step-by-step" cues and
optimize for terse output. Implication: Condition A's prompt cannot
just say "think step-by-step" — needs explicit format scaffolding.

### Explicit-format prompt
Prompt:
```
Format your response EXACTLY like this:
Reasoning step 1: <your first reasoning step>
Reasoning step 2: <your second reasoning step>
...
Final answer: <single letter A, B, C, or D>
```

Output (qwen2.5:7b on a representative test question):
```
Reasoning step 1: The resident is required to maintain accurate and complete medical records...
Reasoning step 2: Ethical standards in medicine mandate transparency with patients regarding their health information...
Reasoning step 3: The attending physician's suggestion to omit the complication...

Final answer: A
```

### Format observations

- **Each step is single-line.** No multi-line wrapping in the
  representative example. Worth verifying across more questions
  during the 3.2 design pass — pathological cases (very long
  reasoning) may wrap.
- **Steps separated by single newlines.** No double-newline between
  steps. Final-answer line preceded by blank line.
- **Step prefix is literal `Reasoning step N:`.** Robust regex:
  `^Reasoning step (\d+):\s*(.*)$`.
- **Final-answer prefix is literal `Final answer:`.** Regex:
  `^Final answer:\s*([A-D])\s*$`. Multi-letter / wrapped variants
  not observed at temp=0; expect more variation at higher temp.
- **Step counts at temp=0:** model produced 3 steps for the
  representative question. Prompt asked "minimum 3, maximum 8" —
  model honored the minimum. Worth checking the distribution at
  scale before locking decomposer's clamp range.
- **Step text length:** 137-213 chars per step in this example.
  Well under e5-large's 512-token window. No chunking concerns
  for step embedding.

### Decomposer parsing strategy implications

1. **Primary path: regex-based extraction.** Two regexes for steps
   and final answer; fast, deterministic, robust to the documented
   prompt format.
2. **Fallback path: JSON-repair prompt** (`prompts/repair.txt`) when
   the regex finds zero steps or no final-answer line. Re-prompt with
   the malformed output and ask the LLM to repair it. CLAUDE.md §7.5
   already anticipates this.
3. **Step-count clamping per CLAUDE.md §7.4:** 3-10 steps; downsample
   evenly-spaced if more, refuse-and-repair if fewer. Default range
   is reasonable per the qwen2.5 sample.
4. **Final-answer parsing is its own concern.** Separate function;
   handle "Final answer: A", "**Answer: A**", "A.", bare "A". The
   exact set depends on real model variance — log unmatched outputs
   during stage 3.3 development to refine the regex.

### Boundary-case finding (empirical validation anchor)

**This is the most consequential single finding from today's
exploration.** qwen2.5:7b at temp=0 produced a confident-sounding
3-step reasoning trajectory and landed on **A** — the wrong answer.
The correct answer is **B** (escalate disagreement with attending
before unilateral disclosure). The model's reasoning was plausible
(documentation completeness, patient transparency, ethics-vs-
instruction) but missed the chain-of-command nuance that's clinically
routine.

This is exactly the boundary case the framework's structural
signature should detect. Confident-but-wrong with clean-looking
reasoning is the failure mode that simple confidence scoring misses
(Condition B would see qwen2.5's "I'm confident in A" and not flag
it). The structural signature, computed against a recovered graph
of MedQA reasoning trajectories, should produce a high score for
this question if the framework works as designed.

**Saved as a future validation anchor at**
`docs/exploration/qwen25_failure_case_2026-05-03.md`. When stage 4
runs, locate this question by `question_id = "medqa-test-0"` and
verify whether the framework's signature score flagged it. That
becomes a concrete validation moment, useful both as a pass/fail
signal for the framework's premise and as a worked example for the
methods paper.

A single observation is not validation — that's stage 4's job. But it
confirms the failure-mode definition is real and present in
qwen2.5:7b at this scale. The signature has something to detect.

## Decomposer failure-mode design (lifted out of "implications" to flag explicitly)

The "Reasoning step N: ..." regex extraction works cleanly on
qwen2.5:7b's structured output at temp=0. But this is the happy path.
What happens when the LLM doesn't comply with the format?

Real-world cases that will surface during stage 3.3 / stage 4:
- LLM gives prose paragraphs without numbered prefixes.
- LLM uses "Step 1:" instead of "Reasoning step 1:".
- LLM uses bullet points or markdown headers.
- LLM gives one long blob with no separation.
- LLM gives a one-letter answer with no reasoning (vanilla "think
  step-by-step" fails this way at temp=0).
- LLM gives reasoning but no "Final answer:" line.

The Decomposer must decide on a behavior for each. Three patterns
with different downstream consequences:

**Strict.** Regex must match the canonical format; non-conforming
output raises `DecomposerError`. The LLMAdapter retry mechanism
(per-item retry, locked in stage 1's protocol) re-issues the call.
Failure-loud, simple, but creates a feedback loop where retries on a
model that consistently misformats produce no progress.

**Graceful with logging.** Try the canonical regex first; on failure,
fall back to heuristic decomposition (split on double-newlines,
treat each paragraph as a step; cap at config.max_steps; refuse if
fewer than config.min_steps). Always produces something; quality
varies; the structural signature reflects the malformedness
(voi_flatness goes high if all "steps" are actually one paragraph;
distance_from_trajectory at later timesteps is undefined if
heuristic produces 1-2 steps).

**Defer to config.** `DecomposerConfig.failure_mode:
Literal["strict", "graceful"]`. Caller picks per scenario (smoke
tests use graceful; gate experiment uses strict).

**My weakly-held lean:** graceful with logging. The framework's
value proposition includes detecting when reasoning is malformed;
the Decomposer producing *some* trajectory even from messy LLM
output gives the structural signature components something to
score, and the score itself reflects the malformedness. The
malformed-output case is in distribution for what the framework
should handle, not an out-of-band failure to escalate.

But strict is defensible — "the LLM didn't follow instructions" is a
different failure mode from "the LLM reasoned poorly," and conflating
them in the signature might dilute the boundary signal.

**Worth the stage 3.2 design pass surfacing this explicitly rather
than letting it emerge as an implementation incident.** The decision
shapes the experiment runner's robustness profile. Recommend the
design pass present the three options with their consequences and
pick deliberately rather than defaulting to one.

## Implications for stage 3.2 design pass

When the design pass starts:

- **Loader is structurally trivial** — HF row to `MedQARawRecord`
  with synthesized question_id and `meta_info` threaded through to
  ground-truth's `secondary_labels["usmle_step"]`. ~30 minutes.
- **Decomposer is the dense work** — regex parsing, fallback to
  repair prompt, step-count clamping, the JSON-malformed edge cases.
  Document expected pathologies. ~2-3 hours design + implementation.
- **Trajectory construction logic** — for live mode, condition C
  iterates over decomposed steps building a multi-state Trajectory.
  For pre-recovered mode, trajectories are loaded from cached
  Parquet. Two distinct flows.

The LLM-test exploration sharpened the decomposer's parsing rules
considerably. The dataset-shape exploration confirmed the loader is
trivial. The design pass's weight should be ~20% loader, ~50%
decomposer, ~30% trajectory construction — as anticipated.
