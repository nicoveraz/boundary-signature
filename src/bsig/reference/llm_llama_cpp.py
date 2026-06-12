"""Reference LLMAdapter for llama.cpp servers (OpenAI-compatible API).

Per ADR-0008, this is the framework's primary measurement adapter under
the unified-measurement protocol. It exposes ``get_token_probabilities``
(reads next-token logprobs at a prompt position, returns the
renormalised conditional distribution plus mass-capture fraction) as
the load-bearing measurement primitive for Condition C.

Design choices (per ADR-0008's design-pass resolutions):

- ``top_k`` is NOT a Protocol parameter. The adapter chooses internally
  based on ``len(token_set)`` and a heuristic (default
  ``max(40, 10 × len(token_set))``). Constructor parameter
  ``logprobs_top_k`` overrides the heuristic at construction time for
  callers with specific model-tokenisation knowledge.

- Mass capture is computed BEFORE renormalisation: ``Σ P(member) for
  member in token_set`` measured against the model's full next-token
  distribution. Returned as part of ``TokenProbabilityResult``.

- Truncation: if any ``token_set`` member is below the adapter's
  effective top-K logprobs, the adapter performs ONE auto-extending
  retry with ``4 × effective_top_k``. If after that retry any member
  still doesn't appear, the adapter populates ``truncated_members`` and
  returns ``P=0`` for those members. This surfaces measurement-quality
  information rather than silently zero-truncating.

- Token aliasing: model tokenisation may emit `` A`` (with leading
  space), ``A`` alone, or ``\\nA`` (with leading newline) for the same
  canonical letter, depending on context. The adapter sums probability
  across these variants when matching ``token_set``.

- Constrained decoding via GBNF grammar is supported as OPTIONAL
  output-safety, NOT measurement strategy. Per the ADR-0008
  investigation, constrained-decoded distributions are mathematically
  equivalent to unconstrained-renormalised at the conditional level
  (KL < 0.0004 nats); the constraint just ensures emitted tokens are
  in ``token_set``. Off by default; enabled via ``output_grammar``
  constructor parameter when callers want predicted-answer output
  safety (e.g., for display).

- Determinism: temperature=0.0 + fixed seed yields ~0.3 % per-letter
  noise across calls on Metal GPU due to floating-point operation
  ordering. Documented in ``get_metadata()`` as
  ``determinism_class = "approximate-fp"``.

"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from bsig.adapters.base import LLMAdapterError
from bsig.adapters.llm import TokenProbabilityResult

if TYPE_CHECKING:
    import httpx


_DEFAULT_HOST = "http://localhost:8080"
_DEFAULT_TIMEOUT = 180.0


class LlamaCppLLMAdapter:
    """LLMAdapter backed by a llama.cpp server's OpenAI-compatible API.

    Constructor parameters:

    - ``model``: identifier for reproducibility logging. The actual
      model loaded by the server is the GGUF passed at server startup;
      this string is recorded in metadata, not used at the wire.
    - ``host``: base URL of the running llama.cpp server.
    - ``timeout``: per-request HTTP timeout, seconds.
    - ``default_seed``: ``seed`` value sent on every call. Combined
      with ``temperature=0.0`` produces approximately reproducible
      outputs (see *determinism* note in module docstring).
    - ``logprobs_top_k``: integer override for the adapter's internal
      top-K heuristic. ``None`` means the adapter computes
      ``max(40, 10 × len(token_set))`` per call. Set explicitly when
      the model's tokenisation is known to produce wider tails.
    - ``output_grammar``: optional GBNF grammar string applied to all
      ``get_token_probabilities`` calls for output-token safety. Does
      NOT change the returned probability distribution (logprobs are
      pre-mask). Default ``None`` (no constraint).
    - ``_client``: optional pre-built ``httpx.Client`` for tests.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        host: str = _DEFAULT_HOST,
        timeout: float = _DEFAULT_TIMEOUT,
        default_seed: int = 42,
        logprobs_top_k: int | None = None,
        output_grammar: str | None = None,
        _client: "httpx.Client | None" = None,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout = timeout
        self._default_seed = default_seed
        self._logprobs_top_k_override = logprobs_top_k
        self._output_grammar = output_grammar
        self._client = _client

    def __del__(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # ---- Text generation ----

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> str:
        last_exc: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                response = self._call(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs_k=None,
                    grammar=None,
                )
                return str(response["text"])
            except Exception as exc:
                last_exc = exc
        raise LLMAdapterError(
            f"LlamaCppLLMAdapter.generate failed after "
            f"{max_retries + 1} attempts (model={self._model!r})"
        ) from last_exc

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Sequence[str]:
        results: list[str] = []
        for i, prompt in enumerate(prompts):
            try:
                results.append(
                    self.generate(prompt, max_tokens, temperature, max_retries)
                )
            except LLMAdapterError as exc:
                partial: list[str | None] = list(results) + [None] * (
                    len(prompts) - len(results)
                )
                raise LLMAdapterError(
                    f"generate_batch failed at index {i}: {exc}",
                    failed_index=i,
                    partial_results=partial,
                ) from exc
        return results

    # ---- Token-probability methods (ADR-0008 primary measurement) ----

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> TokenProbabilityResult:
        if not token_set:
            raise ValueError("token_set must be non-empty")
        unique_tokens = list(dict.fromkeys(token_set))  # preserve order
        effective_top_k = self._effective_top_k(len(unique_tokens))

        last_exc: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                # First attempt at the heuristic top-K
                response = self._call(
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs_k=effective_top_k,
                    grammar=self._output_grammar,
                )
                final_top_logprobs = response["top_logprobs"]
                token_probs = _extract_token_probs(
                    final_top_logprobs, unique_tokens
                )
                missing = [t for t in unique_tokens if t not in token_probs]

                # Auto-extending retry once if any member missing
                if missing:
                    extended_k = effective_top_k * 4
                    response_ext = self._call(
                        prompt=prompt,
                        max_tokens=1,
                        temperature=0.0,
                        logprobs_k=extended_k,
                        grammar=self._output_grammar,
                    )
                    # Extended-K response takes precedence for ALL outputs:
                    # token_probs (computation) and top_k_logprobs (raw
                    # measurement). Consistency: the recorded measurement
                    # corresponds to the response actually used downstream.
                    final_top_logprobs = response_ext["top_logprobs"]
                    token_probs = _extract_token_probs(
                        final_top_logprobs, unique_tokens
                    )
                    missing = [t for t in unique_tokens if t not in token_probs]

                # Build the result
                truncated = tuple(missing)
                # Fill missing members with P=0 (truncation-explicit)
                for t in missing:
                    token_probs[t] = 0.0

                mass = sum(token_probs[t] for t in unique_tokens)
                if mass > 0:
                    distribution = {t: token_probs[t] / mass for t in unique_tokens}
                else:
                    # All members had zero probability — degenerate case
                    raise LLMAdapterError(
                        f"All token_set members had zero probability at "
                        f"this prompt position. token_set={list(token_set)!r}"
                    )

                # Schema-v3: preserve full top-K logprobs as raw measurement.
                # Keys are emitted-token strings (including leading-space and
                # leading-newline aliases as the model emitted them); values
                # are log-probabilities. Downstream consumers exp() to get
                # probabilities; aliasing is downstream's concern (already
                # handled inside this method via _extract_token_probs).
                top_k_logprobs = {
                    entry["token"]: float(entry["logprob"])
                    for entry in final_top_logprobs
                }

                # Schema-v4 (ADR-0009): populate per-position uncertainty
                # signals from the top-K logprobs. Pure derivations; cheap.
                v4_fields = _compute_v4_fields(top_k_logprobs, distribution)

                return TokenProbabilityResult(
                    distribution=distribution,
                    mass_capture=float(mass),
                    truncated_members=truncated,
                    top_k_logprobs=top_k_logprobs,
                    **v4_fields,
                )
            except LLMAdapterError:
                raise  # already final; don't retry on a logical failure
            except Exception as exc:
                last_exc = exc
        raise LLMAdapterError(
            f"LlamaCppLLMAdapter.get_token_probabilities failed after "
            f"{max_retries + 1} attempts (model={self._model!r})"
        ) from last_exc

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> Sequence[TokenProbabilityResult]:
        results: list[TokenProbabilityResult] = []
        for i, prompt in enumerate(prompts):
            try:
                results.append(
                    self.get_token_probabilities(prompt, token_set, max_retries)
                )
            except LLMAdapterError as exc:
                partial: list[TokenProbabilityResult | None] = list(results) + [
                    None
                ] * (len(prompts) - len(results))
                raise LLMAdapterError(
                    f"get_token_probabilities_batch failed at index {i}: {exc}",
                    failed_index=i,
                    partial_results=partial,
                ) from exc
        return results


    # ---- Metadata ----

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "adapter_name": "LlamaCppLLMAdapter",
            "adapter_version": "1",
            "model": self._model,
            "host": self._host,
            "logprobs_top_k_override": (
                str(self._logprobs_top_k_override)
                if self._logprobs_top_k_override is not None
                else "auto"
            ),
            "output_grammar": "set" if self._output_grammar else "none",
            "determinism_class": "approximate-fp",
        }

    # ---- Internal ----

    def _effective_top_k(self, n_tokens: int) -> int:
        if self._logprobs_top_k_override is not None:
            return self._logprobs_top_k_override
        return max(40, 10 * n_tokens)

    def _get_client(self) -> "httpx.Client":
        if self._client is None:
            try:
                import httpx  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "LlamaCppLLMAdapter requires httpx. Install with: "
                    "uv pip install -e '.[llama_cpp]'"
                ) from exc
            self._client = httpx.Client(
                timeout=self._timeout, base_url=self._host
            )
        return self._client

    def _call(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float,
        logprobs_k: int | None,
        grammar: str | None,
    ) -> dict[str, Any]:
        """Single HTTP call to llama.cpp /v1/completions.

        Returns a dict with:
        - "text": the emitted text (string)
        - "top_logprobs": list of {"token": str, "logprob": float} dicts
          at the first generated position; empty if logprobs_k is None
        """
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "LlamaCppLLMAdapter requires httpx. Install with: "
                "uv pip install -e '.[llama_cpp]'"
            ) from exc

        client = self._get_client()
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens is not None else 256,
            "temperature": temperature,
            "seed": self._default_seed,
        }
        if logprobs_k is not None:
            payload["logprobs"] = logprobs_k
        if grammar is not None:
            payload["grammar"] = grammar

        response = client.post("/v1/completions", json=payload)
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"llama.cpp returned non-JSON: {response.text[:200]}"
            ) from exc

        choices = data.get("choices")
        if not choices:
            raise KeyError(
                f"llama.cpp response missing 'choices': {response.text[:200]}"
            )
        choice = choices[0]
        text = choice.get("text", "")

        top_logprobs: list[dict[str, Any]] = []
        if logprobs_k is not None:
            logprobs_obj = choice.get("logprobs")
            if logprobs_obj is None:
                raise KeyError(
                    "llama.cpp response missing 'logprobs' despite request"
                )
            content = logprobs_obj.get("content")
            if content:
                # OpenAI-compatible format: content is a list of token-positions;
                # each position has top_logprobs (list of {token, logprob, ...}).
                first_position = content[0]
                top_logprobs = first_position.get("top_logprobs", [])

        return {"text": text, "top_logprobs": top_logprobs}


def _extract_token_probs(
    top_logprobs: list[dict[str, Any]],
    token_set: Sequence[str],
) -> dict[str, float]:
    """Extract per-token probability from a top_logprobs response,
    summing probability across leading-space and leading-newline aliases.

    Returns a dict mapping each present member of token_set to its
    summed probability. Missing members (no variant in top_logprobs)
    are absent from the returned dict.
    """
    by_emitted_token: dict[str, float] = {
        entry["token"]: math.exp(entry["logprob"])
        for entry in top_logprobs
    }
    result: dict[str, float] = {}
    for canonical in token_set:
        prob = 0.0
        found = False
        for variant in (canonical, " " + canonical, "\n" + canonical):
            if variant in by_emitted_token:
                prob += by_emitted_token[variant]
                found = True
        if found:
            result[canonical] = prob
    return result


def _compute_v4_fields(
    top_k_logprobs: Mapping[str, float],
    distribution: Mapping[str, float],
) -> dict[str, float | None]:
    """Compute schema-v4 per-position uncertainty fields from the
    cached top-K logprobs and renormalised distribution. Pure
    derivation; cheap. Returns a dict suitable for splatting into
    TokenProbabilityResult kwargs.

    Per ADR-0009: each field is optional; values populated when
    derivable from the available data, None otherwise.
    """
    if not top_k_logprobs:
        return {}
    # Reuse the framework's signature scorer functions for consistency
    # — the per-position scorers in core/signature.py operate on the
    # same Mapping[str, float] shape.
    from bsig.core.signature import (  # noqa: PLC0415
        entropy_full_from_top_k,
        gap_top1_top_k_from_top_k,
        gap_top2_from_top_k,
        p_max_from_top_k,
        top_k_mass_from_top_k,
    )

    # Entropy of hypothesis-space distribution (in nats; renormalised
    # over distribution.keys())
    h_hyp = 0.0
    for p in distribution.values():
        if p > 0:
            h_hyp -= p * math.log(p)

    # Chosen logprob: argmax over the full top-K (the token the model
    # would emit if temperature=0)
    chosen_logprob = max(top_k_logprobs.values())

    return {
        "p_max": p_max_from_top_k(top_k_logprobs),
        "entropy_full": entropy_full_from_top_k(top_k_logprobs),
        "entropy_hyp": h_hyp,
        "top_k_mass": top_k_mass_from_top_k(top_k_logprobs, k=10),
        "gap_top2": gap_top2_from_top_k(top_k_logprobs),
        "gap_top1_topK": gap_top1_top_k_from_top_k(top_k_logprobs, k=10),
        "chosen_logprob": chosen_logprob,
    }
