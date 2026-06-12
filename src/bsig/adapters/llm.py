"""LLMAdapter protocol.

Wraps any language model. Provides two families of operations:

- **Text generation** (``generate`` / ``generate_batch``): raw
  completion-style text output. Used by Conditions A, B, and the
  initial CoT step of Condition C.
- **Token-probability queries** (``get_token_probabilities`` /
  ``get_token_probabilities_batch``): direct measurement of the
  model's next-token distribution at a prompt position, renormalised
  over a finite token set, with the mass-capture fraction returned
  alongside. Per ADR-0008 this is the framework's primary measurement
  primitive for Condition C's per-step monitoring under the unified-
  measurement protocol.

Protocol history: this surface originally exposed verbalised-
distribution methods (``get_hypothesis_distribution`` /
``get_hypothesis_distribution_batch``) as the only measurement path.
The text-generation methods were added during stage 3.3a (see
ADR-0005) when Conditions A and B surfaced the asymmetry. The
token-probability methods were added during the stage-4a follow-up
(see ADR-0008) when the verbalised-distribution methodology was
abandoned. The verbalised-distribution methods were removed from the
Protocol in 2026-05-05 cleanup once no consumer remained — see
ADR-0008 §"Cleanup".
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TokenProbabilityResult:
    """Output of ``get_token_probabilities``.

    Four fields (two required, two with defaults). Per ADR-0008 (v1) and
    the schema-v3 measurement-vs-computation distinction (post-stage-4a):

    - ``distribution``: the renormalised conditional over the requested
      ``token_set``. Keys are exactly ``set(token_set)``; values sum to
      1.0 ± 1e-6. Members with negligible model probability appear with
      value 0.0 (and in ``truncated_members``).
    - ``mass_capture``: the fraction of next-token mass that landed on
      ``token_set`` before renormalisation. In ``[0.0, 1.0]``.
    - ``truncated_members``: tuple of ``token_set`` members below the
      adapter's effective top-K even after one auto-extending retry.
      Empty in the common case.
    - ``top_k_logprobs``: the full top-K logprobs returned by the adapter
      at this measurement position, as a mapping of *every* emitted token
      (not just ``token_set`` members) to its log-probability. Empty
      mapping by default for adapters that cannot expose this. Stored as
      raw measurement (per the project's
      *measurement vs computation* methodology — preserved at full
      fidelity so downstream computations can be re-derived without
      re-running inference). Storage cost: ~5-10 KB per measurement
      position; cumulative ~9 MB at N=1273. Cached-trajectories schema-v3
      preserves these; v2 trajectories load with empty mapping.
    """

    distribution: Mapping[str, float]
    mass_capture: float
    truncated_members: tuple[str, ...] = field(default_factory=tuple)
    top_k_logprobs: Mapping[str, float] = field(default_factory=dict)

    # Schema-v4 additive fields (ADR-0009). All optional; default to None
    # for backward compat with v2/v3 cached trajectories. Adapters that can
    # compute them populate; adapters that cannot leave None.
    p_max: float | None = None
    entropy_full: float | None = None  # nats; from top-K + lump-residual
    entropy_hyp: float | None = None  # nats; entropy of `distribution`
    top_k_mass: float | None = None  # cumulative top-10 mass
    gap_top2: float | None = None  # P(top-1) - P(top-2) over top-K
    gap_top1_topK: float | None = None  # P(top-1) - P(K-th) over top-K
    chosen_logprob: float | None = None  # logprob of emitted token (nats)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Output of ``generate_completions_with_shared_prefix``.

    Per stage-6 MLX adapter pre-design (commit ``4cf2c04``): semantic
    entropy and temperature-varied comparison need N-sample completions
    sharing a single prefix-prefill. Each completion carries text,
    raw token IDs, and per-step chosen-token log-probabilities (sufficient
    for length-normalised sequence probability used in cluster-probability
    weighting).

    - ``text``: decoded completion (excluding the prefix).
    - ``tokens``: raw token IDs of the completion (excluding the prefix).
    - ``chosen_logprobs``: log-probability of each emitted token in
      ``tokens``. Length matches ``tokens``. ``sum / N`` gives length-
      normalised mean log-probability for cluster weighting.
    - ``stop_reason``: one of ``"max_tokens"``, ``"stop_token"``,
      ``"unknown"``. Useful for cluster-quality diagnostics.
    """

    text: str
    tokens: tuple[int, ...]
    chosen_logprobs: tuple[float, ...]
    stop_reason: str = "unknown"


class LLMAdapter(Protocol):
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> str:
        """Return raw text completion for ``prompt``.

        ``max_tokens=None`` lets the implementation choose (typically
        the model's natural max). ``temperature=0.0`` is the
        recommended default for deterministic Conditions A/B/C runs;
        higher temperatures are valid for sampling-based studies.

        ``max_retries`` is the number of retry attempts on transport
        failures before raising ``LLMAdapterError``. Per-call
        semantics; ``generate`` does not have parse failures (raw text
        is always returned), so retries cover only network/server
        issues.
        """
        ...

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Sequence[str]:
        """Return raw text completions for each prompt in order.

        Retry semantics are per-item: each element gets up to
        ``max_retries`` independent retries on transport failures.
        Successful items must not be re-issued when other items fail
        or retry, preserving reproducibility and avoiding wasted
        compute on partial failures.

        Returns a sequence of length ``len(prompts)``, in the same
        order. Items that exhausted retries raise ``LLMAdapterError``;
        partial-success behavior is not supported (atomic batch:
        either all completions or an exception with the successful
        items discarded).
        """
        ...

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> TokenProbabilityResult:
        """Read the model's next-token distribution at the position
        immediately following ``prompt``, renormalised over
        ``token_set``.

        Returns a :class:`TokenProbabilityResult` carrying the
        renormalised conditional distribution, the mass-capture fraction
        (Σ pre-renormalisation probability over ``token_set``), and any
        ``truncated_members`` (members below the adapter's effective
        top-K even after one auto-extending retry).

        Implementations choose top-K internally based on
        ``len(token_set)`` and model-tokenization properties (see
        ADR-0008 for the design-pass rationale). A reasonable default
        heuristic is ``effective_top_k = max(40, 10 × len(token_set))``.
        Token-aliasing (e.g. ``" A"`` vs ``"A"`` vs ``"\\nA"``) is
        handled by the implementation; the returned distribution's keys
        are exactly ``set(token_set)``.

        Adapters that lack logprobs API access (e.g. Anthropic API,
        Ollama at the time of writing) cannot meaningfully implement
        this method; they should raise ``LLMAdapterError`` with an
        actionable message naming an adapter that does support it.

        ``max_retries`` is per-call retries on transport failures.
        Raises ``LLMAdapterError`` on exhausted retries.
        """
        ...

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> Sequence[TokenProbabilityResult]:
        """Return a :class:`TokenProbabilityResult` for each prompt in
        order.

        Per-item retry semantics: each element gets up to
        ``max_retries`` independent retries on transport failures;
        successful items must not be re-issued when other items fail
        or retry. Implementations are encouraged (not required) to
        populate ``failed_index`` and ``partial_results`` on raised
        :class:`LLMAdapterError` for surgical-repair support
        downstream. Implementations that can't preserve partial state
        get atomic-repair fallback in the caller.

        Returns a sequence of length ``len(prompts)``, in the same order.
        Items that exhausted retries raise ``LLMAdapterError``.
        """
        ...

    def get_metadata(self) -> Mapping[str, str]:
        """Model name, revision, version for reproducibility logging."""
        ...
