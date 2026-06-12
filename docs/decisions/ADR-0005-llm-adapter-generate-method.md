# ADR-0005: LLMAdapter Protocol extended with text-generation methods

**Status:** Accepted
**Date:** 2026-05-03
**Stage of origin:** 3.3a design pass

## Context

The stage-1 ``LLMAdapter`` Protocol exposed only hypothesis-distribution
methods (``get_hypothesis_distribution`` and
``get_hypothesis_distribution_batch``). At the time, Condition C was
the only anticipated consumer, and its per-step distribution monitoring
was the dominant operational need.

Stage 3.3a (Conditions A and B) surfaced an asymmetry: A and B want
the LLM's reasoning text plus a single answer (A) or answer +
confidence (B). Forcing this through ``get_hypothesis_distribution``
produces awkward implementations that hide the actual operation
(text generation) behind a distribution-shaped wrapper.

## What was hidden behind the original Protocol

For an instruction-tuned model behind an OpenAI-compatible API,
``get_hypothesis_distribution`` is *implemented* via text generation:
the implementation prompts the model with a structured request, the
model generates text, the parser extracts a probability distribution.
Generation IS the underlying operation; distribution extraction is the
post-processing.

Three consequences for Conditions A and B:
- They want the text PLUS something else (answer letter for A,
  confidence for B). The "something else" parsing is condition-
  specific, not generic distribution extraction.
- Routing through ``get_hypothesis_distribution`` would mean each
  condition's adapter implementation sets up a distribution-extraction
  pipeline only to throw most of the result away.
- The hypothesis-space concept doesn't apply cleanly to A/B's
  "produce reasoning and an answer" task — they're not querying a
  distribution, they're generating text.

## Decision

Add ``generate`` and ``generate_batch`` to the Protocol alongside the
existing methods. The Protocol now has five methods:

```python
class LLMAdapter(Protocol):
    def generate(...) -> str: ...
    def generate_batch(...) -> Sequence[str]: ...
    def get_hypothesis_distribution(...) -> Mapping[str, float]: ...
    def get_hypothesis_distribution_batch(...) -> Sequence[Mapping[str, float]]: ...
    def get_metadata(...) -> Mapping[str, str]: ...
```

Implementations may build ``get_hypothesis_distribution`` on top of
``generate`` (e.g., the OpenAI-compatible adapter generates a
structured response and parses it) or implement them independently
(e.g., an adapter that uses logit-extraction for distributions and
chat-completion for generation).

## Why this is strictly additive

- Existing test mocks gain two methods to provide.
- No existing caller of ``get_hypothesis_distribution`` is affected.
- The Protocol surface grows but the semantics of existing methods
  do not change.

## Why this is the right framing (not just a quick fix)

The original Protocol design was right for what stage 1 knew:
Condition C was the only conceptual consumer, and distributions were
the operationally relevant output. Stage 3.3 produced new information
(Conditions A and B exist; they need text, not distributions),
warranting a deliberate Protocol evolution.

Hiding generation behind distribution-only made sense when the only
consumer was Condition C; with three conditions in the picture the
asymmetry is no longer defensible. The Protocol should be honest about
what it offers.

## Retry semantics for the new methods

``generate`` retries on transport failures only — there are no parse
failures because raw text is always returned.

``generate_batch`` follows the same per-item retry semantics as
``get_hypothesis_distribution_batch`` (locked stage 1): each element
gets up to ``max_retries`` independent retries on transport failures.
Successful items must not be re-issued when other items fail or
retry. Partial-success not supported (atomic batch: either all
completions or an exception with successful items discarded).

## Forward implications

- Stage 3.3a (Conditions A, B) consumes ``generate`` directly.
- Stage 3.3b (Condition C) uses ``generate`` for the initial CoT and
  ``get_hypothesis_distribution`` for per-step queries.
- Stage 3.5 (reference adapter implementations) implements all five
  methods. ``llm_local`` (Ollama) maps both pairs to Ollama's
  generate endpoint with different prompt wrappers. ``llm_openai``
  (vLLM-compatible) maps generate to chat completion and
  hypothesis-distribution to a structured-output prompt + parser.
- Future ``BaseLLMAdapter`` (deferred per ADR-0004) implements
  ``generate`` via few-shot continuation and
  ``get_hypothesis_distribution`` via softmax over answer-position
  logits. The two paths are operationally different but both satisfy
  the same Protocol.
