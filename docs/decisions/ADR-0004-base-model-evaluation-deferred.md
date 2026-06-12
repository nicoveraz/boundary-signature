# ADR-0004: Base-model evaluation deferred to post-gate

**Status:** Deferred
**Date:** 2026-05-03
**Stage of origin:** Pre-3.2 grounded exploration

Stage 3.2's grounded exploration tested `qwen2.5:7b-instruct` — an
instruction-tuned model with RLHF. The framework's `LLMAdapter`
Protocol from stage 1 is generic enough to support **base models**
(raw pretrained, no RLHF) as well, but the adapter implementation
differs substantially across the two paradigms.

## The two paradigms

**Instruction-tuned implementation** (current `LLMAdapter` reference
work, stage 3.5):
- Constructs chat-style prompts with explicit format scaffolding
  ("Reasoning step N: ...", "Final answer: ...").
- Parses the model's text response with regex extraction.
- Asks the model to express confidence; reads it from response text.
- Has format-compliance failures; per-item retry semantics
  (locked in stage 1's protocol) handle these.

**Base-model implementation** (deferred):
- Constructs few-shot prompts with 3-8 worked examples ending at
  the answer position.
- Runs forward inference, extracts logits at the answer position,
  softmax-normalizes over hypothesis-space tokens.
- Hypothesis distribution is "what tokens the model assigns
  probability mass to" — not "what the model says it believes."
- No format-compliance failures (forward inference always returns
  logits); requires raw model access (logit inspection), not just
  a chat-completions API. vLLM exposes this; OpenAI does not;
  Ollama partially does.

These are formally different objects. For an instruction-tuned model,
hypothesis_distribution reflects what the model is "trying to
express." For a base model, it reflects what the model "knows" in a
more raw sense. The framework's structural signatures may behave
differently on each (in particular, base-model `entropy_plateau` is
likely lower on average — token preferences are sharper than
expressed-confidence distributions).

## Decision

Defer base-model evaluation to post-gate work. The chest-pain gate
experiment evaluates instruction-tuned models only.

If the gate passes, cross-paradigm evaluation becomes a natural
**Paper 2** contribution per charter §9's publication strategy:
"structural signatures across model paradigms" is meaningfully
stronger evidence that the signatures detect something fundamental
about reasoning rather than artifacts of instruction-following.

## Architectural impact: none

The `LLMAdapter` Protocol from stage 1 supports both paradigms. The
contract is generic enough that a `BaseLLMAdapter` implementation
satisfies it without protocol changes. Stage 3.5's reference work
builds a single instruction-tuned implementation
(`bsig.reference.llm_local` for Ollama smoke; `bsig.reference.llm_openai`
for vLLM at H100); a `bsig.reference.llm_base` is added later if
needed.

## Reasons for deferral

- **Gate experiment scope.** The chest-pain gate passes or fails on
  instruction-tuned models alone (AUC ≥ 0.65 vs the structural
  signature). Adding base-model evaluation expands the experimental
  matrix without changing the primary outcome.
- **Implementation cost.** Stage 3.5 expands from "one reference
  adapter for Ollama smoke" to "two adapters with structurally
  different operational paradigms" — at least a doubling of stage
  3.5's scope.
- **Compute budget.** Logit extraction requires the full hidden state
  per inference, costing more memory than chat completion. The H100
  $50 ceiling tightens.
- **Post-gate framing is naturally rich.** Cross-paradigm evaluation
  is a strong Paper 2 motif rather than a gate-experiment add-on.

## Implications for current work

- **Stage 3.2 design pass:** no change required. Decomposer's
  graceful-failure-mode design (per pre-design notes) becomes more
  important if base models are a future target — base-model output
  doesn't follow the "Reasoning step N: ..." format and the
  Decomposer needs to handle arbitrary continuation text. This is
  one more reason graceful-with-logging is the right lean.
- **MCQStateCanonicalizer:** unchanged. Hashes reasoning content;
  works regardless of which model produced it.
- **Stage 3.5:** scoped to instruction-tuned reference adapter
  (Ollama + OpenAI/vLLM-compatible); base-model adapter explicitly
  out of scope.
- **Stage 4 H100 run:** instruction-tuned models only.
- **Stage 6 chest-pain gate:** instruction-tuned models only.
- **Hypothetical post-gate Paper 2:** revisits this ADR; adds
  `bsig.reference.llm_base` (or equivalent); re-runs against same
  graphs with base-model trajectories; compares signature behavior
  across paradigms.
