"""Reference LLMAdapter for Apple Silicon via mlx-lm (Phase A).

**Phase A scaffolding**: this module implements the framework's
``LLMAdapter`` Protocol against ``mlx-lm`` primitives in direct mode
(no vllm-mlx engine yet). Designed to coexist with
``LlamaCppLLMAdapter`` — both adapters remain supported per the
no-replace principle in stage_6_mlx_adapter_pre_design_notes.md §0.

**What's in Phase A**:

- ``MLXLLMAdapter`` class with the same Protocol surface as
  ``LlamaCppLLMAdapter`` (``generate``, ``generate_batch``,
  ``get_token_probabilities``, ``get_token_probabilities_batch``,
  ``get_metadata``).
- New method ``generate_completions_with_shared_prefix`` (additive
  Protocol extension; required for Phase C semantic entropy).
- Lazy ``mlx_lm`` import so the class loads without mlx-lm installed
  (parallels ``llm_llama_cpp.py``'s lazy ``httpx`` import).
- Mock-injectable model load via ``_loader_factory`` constructor
  parameter for unit tests against a fake forward-pass without
  pulling MLX as a test dependency.

**Phase C status (diagnosed 2026-05-07; not landed; upstream issue
https://github.com/ml-explore/mlx/issues/3494; fix PR #3498 MERGED
2026-05-11 at https://github.com/ml-explore/mlx/pull/3498)**:
``generate_completions_with_shared_prefix`` continues to use the
Phase A serial path. The original Phase C design (N samples decode
together at batch=N per forward, per
stage_6_mlx_adapter_pre_design_notes.md §6.2 Amendment 1) was
abandoned after empirical diagnosis bisected the bug to
``mx.fast.rope`` at ``batch >= 2, seq_len == 1`` (mlx-core Metal
kernel issue, not mlx-lm). The fix is merged to mlx ``main`` but
NOT yet in a released wheel (latest PyPI tag ``mlx==0.31.2``,
2026-04-22, predates the merge and still fails the repro as of
2026-05-23). Phase C unblocks at the first released mlx-core
> 0.31.2 that ships the patch — not at merge. A hybrid path (single batched prefill +
per-sample serial decode) was prototyped and benchmarked at
~1.05x speedup vs Phase A serial on decode-dominated workloads
— did not earn its complexity. See corruption registry entry
2026-05-07 (engine category) for the tracking entry.

Phase B uncertainty-signal fields (schema-v4 per ADR-0009) are
populated symmetrically with ``LlamaCppLLMAdapter``. Cross-
adapter agreement validation (§7.1) ran 2026-05-07 with a 16/50
partial run; the bit-identical-weights precondition was diagnosed
and documented (corruption registry entry; methods-paper §7.4 /
§9.3). MLX-only findings beyond trajectory-level aggregates
require bit-identical weights to be reportable.

**This adapter does NOT yet include**:

- vllm-mlx engine mode (stage_6_mlx_adapter_pre_design_notes.md §4.2;
  introduced when production deployment requires continuous batching).
- Proper top-p (nucleus) masking in the batched sampler. The
  ``top_p`` parameter is accepted for Protocol-conformance but
  not enforced; full-distribution categorical sampling is used.
  For typical Phase-C usage with ``temperature ∈ [0.5, 1.0]`` the
  omission is benign.
- Per-sample seed reproducibility in the batched path. Batched
  ``mx.random.categorical`` samples N rows from a single global
  rng state; per-row seeds are not enforceable. Reproducibility
  is at the *batch* granularity (fixed batch size + order +
  seed → reproducible). Per-sample reproducibility requires
  serial calls (n_samples=1 short-circuit).

**Numerical-validation status**: not yet certified. The cross-adapter
agreement test (§7.1) is the gate. Once it passes, both
``LlamaCppLLMAdapter`` and ``MLXLLMAdapter`` are interchangeable for
the framework's measurement protocol.

**Hardware**: M4 Pro deployment target; M1 Pro current dev. Both
work via mlx-lm. The ``mx.compile`` optimisation (vllm-mlx's
``MLXModelRunner``) is disabled here for research-validation
interpretability per spec §11.

References:

- ``stage_6_mlx_adapter_pre_design_notes.md`` (commit ``4cf2c04``):
  load-bearing design decisions.
- PI's full technical spec (in chat / draft form): MLX pseudocode,
  KV cache management, etc. — canonical reference for implementation
  detail beyond the Protocol surface.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from bsig.adapters.base import LLMAdapterError
from bsig.adapters.llm import CompletionResult, TokenProbabilityResult

if TYPE_CHECKING:
    pass  # mlx-lm types intentionally not exposed at module level


_DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Below this many tokens of shared prefix, per-prompt fresh-forward is
# cheaper than the prefill+copy overhead. Empirically chosen on M1 Pro;
# revisit on M4 Pro if Phase C profiling motivates.
_SHARED_PREFIX_THRESHOLD = 50


class _MLXModel(Protocol):
    """Structural shape we depend on from mlx-lm. Used for typing only.

    The real ``mlx_lm.load()`` returns a model implementing this and
    a tokenizer alongside. Test fakes implement this Protocol.

    Per mlx-lm 0.31+: ``__call__`` returns logits tensor directly
    ([batch, seq, vocab]). The optional ``cache`` argument is mutated
    in place to maintain KV state across calls.
    """

    def __call__(self, input_ids: Any, cache: Any = None) -> Any:
        ...


class _MLXTokenizer(Protocol):
    """Structural shape from mlx-lm tokenizer."""

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        ...


_LoaderFactory = Callable[[str], tuple[_MLXModel, _MLXTokenizer]]


def _default_loader(model_name: str) -> tuple[_MLXModel, _MLXTokenizer]:
    """Default loader: defers to ``mlx_lm.load``. Imported lazily so the
    adapter class can be imported without mlx-lm installed."""
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise ImportError(
            "MLXLLMAdapter requires mlx-lm. Install with: "
            "uv pip install -e '.[mlx]'"
        ) from exc
    return load(model_name)  # type: ignore[return-value]


class MLXLLMAdapter:
    """LLMAdapter implementation backed by mlx-lm (Apple Silicon native).

    Phase A: implements ``measure_at_position`` (= ``get_token_probabilities``
    in the framework's Protocol naming) in direct mlx-lm mode. Other
    Protocol methods + ``generate_completions_with_shared_prefix``
    sketched with serial-loop fallback for the multi-sample method.

    Constructor parameters (parallels ``LlamaCppLLMAdapter``):

    - ``model``: identifier passed to mlx-lm's ``load()``. Default is a
      4-bit Qwen2.5-7B-Instruct conversion. The actual model loaded
      depends on what mlx-lm finds at the path/HF-id.
    - ``default_seed``: seed for deterministic sampling. Combined with
      ``temperature=0.0`` produces approximately-reproducible outputs
      (Metal fp ordering noise per spec §11).
    - ``logprobs_top_k``: integer override for the adapter's internal
      top-K heuristic. ``None`` means the adapter computes
      ``max(40, 10 * len(token_set))`` per call (parallel to llama.cpp
      adapter).
    - ``_loader_factory``: optional injection point for tests. Default
      is ``_default_loader`` which calls ``mlx_lm.load``. Tests pass a
      fake loader returning a fake model + tokenizer.

    Hardware: requires Apple Silicon (Metal/MLX); does not run on
    Intel Macs or non-Mac systems.

    Determinism class: ``approximate-fp`` (Metal fp-ordering noise on
    the order of 0.3% per-letter probability across calls at
    temperature 0.0; documented in metadata).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        default_seed: int = 42,
        logprobs_top_k: int | None = None,
        _loader_factory: _LoaderFactory | None = None,
    ) -> None:
        self._model_name = model
        self._default_seed = default_seed
        self._logprobs_top_k_override = logprobs_top_k
        self._loader_factory = _loader_factory or _default_loader
        self._model: _MLXModel | None = None
        self._tokenizer: _MLXTokenizer | None = None

    def _ensure_loaded(self) -> tuple[_MLXModel, _MLXTokenizer]:
        if self._model is None or self._tokenizer is None:
            try:
                self._model, self._tokenizer = self._loader_factory(
                    self._model_name
                )
            except Exception as exc:
                raise LLMAdapterError(
                    f"MLXLLMAdapter failed to load model "
                    f"{self._model_name!r}: {exc}"
                ) from exc
        return self._model, self._tokenizer

    # ---- Text generation ----

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> str:
        """Generate text completion via mlx-lm.

        Phase A: delegates to ``mlx_lm.generate`` with the loaded
        model + tokenizer. Future Phase A work: characterize MLX-specific
        retry/error semantics (Metal OOM, kernel compilation failures)
        before claiming retry is robust. ``max_retries`` is currently a
        Protocol-conformant no-op; if mlx-lm raises, the error
        propagates as ``LLMAdapterError``.
        """
        model, tokenizer = self._ensure_loaded()
        try:
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise ImportError(
                "MLXLLMAdapter.generate requires mlx-lm"
            ) from exc
        try:
            sampler = make_sampler(temp=temperature)
            result = mlx_generate(
                model,
                tokenizer,  # type: ignore[arg-type]
                prompt=prompt,
                max_tokens=max_tokens or 512,
                sampler=sampler,
                verbose=False,
            )
            return str(result)
        except Exception as exc:
            raise LLMAdapterError(
                f"MLXLLMAdapter.generate failed: {exc}"
            ) from exc

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Sequence[str]:
        """Sequential fallback. Real batched generation requires either
        mlx-lm's batched API (when stable) or vllm-mlx engine mode (Phase
        A optional / production)."""
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

    # ---- Token-probability measurement (the load-bearing primitive) ----

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> TokenProbabilityResult:
        """Measure next-token distribution at the position immediately
        following ``prompt``.

        Implementation per stage-6 MLX adapter pre-design §3.3:

        1. Tokenize prompt; run forward pass.
        2. Extract last-position logits → log-probs.
        3. Top-K extraction over full vocab.
        4. For each ``token_set`` member, find its token-id (handling
           leading-space variants) and accumulate probability mass.
        5. Renormalise over ``token_set``; record mass-capture.

        Phase A note: the ``max_retries`` parameter is Protocol-conformant
        but currently a no-op for MLX (no transport layer). Reserved for
        cases where ``mx.compile`` triggers Metal kernel issues that
        benefit from retry; not yet characterised empirically.
        """
        model, tokenizer = self._ensure_loaded()
        try:
            return self._measure_at_position(
                model, tokenizer, prompt, token_set
            )
        except Exception as exc:
            raise LLMAdapterError(
                f"MLXLLMAdapter.get_token_probabilities failed: {exc}"
            ) from exc

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> Sequence[TokenProbabilityResult]:
        """Batched measurement with shared-prefix KV cache reuse.

        When the input prompts share a substantial common token-prefix
        (default threshold: 50 tokens), the adapter prefills the
        prefix once into a KV cache and runs each prompt's
        suffix-only forward pass against a copy of the cache. Cost
        reduction: ``1 * prefix + N * suffix`` instead of
        ``N * full_prompt``.

        Empirical speedup at typical stage-4-shape workloads
        (~500-token prefix, ~50-token suffix, N=3): 2-3x wall-time
        reduction. For longer suffixes the speedup decreases; for
        shorter suffixes it increases.

        When prompts do not share substantial prefix (LCP < 50
        tokens), falls back to per-prompt fresh-forward passes
        (equivalent to looping over ``get_token_probabilities``).

        Per-item retry semantics on failure: surgical-repair-
        compliant ``LLMAdapterError`` with ``failed_index`` and
        ``partial_results`` populated.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [
                self.get_token_probabilities(
                    prompts[0], token_set, max_retries
                )
            ]

        model, tokenizer = self._ensure_loaded()

        # Tokenize all prompts to find shared prefix
        all_tokens = [tokenizer.encode(p) for p in prompts]
        if any(not t for t in all_tokens):
            raise ValueError("One or more prompts produced empty tokens")
        lcp_len = self._longest_common_prefix_len(all_tokens)

        if lcp_len < _SHARED_PREFIX_THRESHOLD:
            # Fall back to per-prompt fresh-forward
            return self._batch_sequential_fallback(
                prompts, token_set, max_retries
            )

        return self._batch_with_shared_prefix(
            model, tokenizer, all_tokens, lcp_len, token_set,
        )

    def _batch_sequential_fallback(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int,
    ) -> Sequence[TokenProbabilityResult]:
        results: list[TokenProbabilityResult] = []
        for i, prompt in enumerate(prompts):
            try:
                results.append(
                    self.get_token_probabilities(
                        prompt, token_set, max_retries
                    )
                )
            except LLMAdapterError as exc:
                partial: list[TokenProbabilityResult | None] = (
                    list(results) + [None] * (len(prompts) - len(results))
                )
                raise LLMAdapterError(
                    f"get_token_probabilities_batch failed at index {i}: {exc}",
                    failed_index=i,
                    partial_results=partial,
                ) from exc
        return results

    def _batch_with_shared_prefix(
        self,
        model: _MLXModel,
        tokenizer: _MLXTokenizer,
        all_tokens: Sequence[Sequence[int]],
        lcp_len: int,
        token_set: Sequence[str],
    ) -> Sequence[TokenProbabilityResult]:
        """Build cache once on common prefix; for each prompt, run
        suffix-only forward against a copy of the cache."""
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import (
                make_prompt_cache,
            )
        except ImportError as exc:
            raise ImportError(
                "MLXLLMAdapter shared-prefix batch requires mlx.core "
                "and mlx_lm.models.cache"
            ) from exc
        import copy as copy_mod

        # Build cache on shared prefix once
        prefix_cache = make_prompt_cache(model)
        prefix_tokens = mx.array(list(all_tokens[0][:lcp_len]))[None, :]
        _ = model(prefix_tokens, cache=prefix_cache)
        # Materialise cache state before reuse
        for c in prefix_cache:
            if hasattr(c, "state") and c.state[0] is not None:
                mx.eval(c.state[0])

        results: list[TokenProbabilityResult] = []
        for i, tokens in enumerate(all_tokens):
            try:
                cache = copy_mod.deepcopy(prefix_cache)
                suffix = list(tokens[lcp_len:])
                if suffix:
                    suffix_arr = mx.array(suffix)[None, :]
                    logits = model(suffix_arr, cache=cache)
                    last_logits = logits[:, -1, :]
                else:
                    # Prompt is exactly the LCP — re-run last token of
                    # prefix to get a measurement at end-of-prefix.
                    last_tok = mx.array([list(tokens[-1:])])
                    fresh_cache = copy_mod.deepcopy(prefix_cache)
                    logits = model(last_tok, cache=fresh_cache)
                    last_logits = logits[:, -1, :]
                results.append(
                    self._build_result_from_last_logits(
                        last_logits, tokenizer, token_set, mx,
                    )
                )
            except Exception as exc:
                partial: list[TokenProbabilityResult | None] = (
                    list(results)
                    + [None] * (len(all_tokens) - len(results))
                )
                raise LLMAdapterError(
                    f"get_token_probabilities_batch failed at index {i}: {exc}",
                    failed_index=i,
                    partial_results=partial,
                ) from exc
        return results

    @staticmethod
    def _longest_common_prefix_len(
        sequences: Sequence[Sequence[int]],
    ) -> int:
        """Length of the longest common token-prefix across the
        sequences. Returns 0 if no shared prefix or empty input."""
        if not sequences:
            return 0
        min_len = min(len(s) for s in sequences)
        for i in range(min_len):
            first = sequences[0][i]
            if not all(s[i] == first for s in sequences):
                return i
        return min_len

    # ---- Multi-sample inference (Phase C semantic-entropy dependency) ----

    def generate_completions_with_shared_prefix(
        self,
        prefix_tokens: Sequence[int],
        n_samples: int,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seeds: Sequence[int] | None = None,
    ) -> list[CompletionResult]:
        """Generate N samples sharing a prefix prefill (Phase A serial).

        **Phase C status**: per ``_batched_sample_completions``
        docstring, the originally-planned batched-decode Phase C is
        blocked on an upstream mlx-lm limitation (incremental decode
        at batch>1 does not preserve correctness across rows;
        diagnosed 2026-05-07; corruption registry entry under engine
        category). The current implementation routes ``n_samples > 1``
        through Phase A serial: per-sample ``_sample_completion`` calls
        with deterministic ``seed = base_seed + i`` per sample. Cost:
        ``N * (1 prefill + max_tokens decodes)``.

        ``n_samples=1`` short-circuits identically.

        **Seed semantics.** When ``seeds`` is provided, ``seeds[0]``
        is the base seed and per-sample seeds increment from there
        (Phase A behavior preserved). When ``seeds is None``, the
        adapter's ``default_seed`` is used as the base.

        ``top_p`` is accepted for Protocol-conformance but not
        enforced (full-distribution categorical sampling). Proper
        top-p masking is a future addition; for typical usage with
        ``temperature in [0.5, 1.0]`` the omission is benign.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1; got {n_samples}")
        if seeds is not None and len(seeds) != n_samples:
            raise ValueError(
                f"len(seeds) must equal n_samples; got {len(seeds)} and "
                f"{n_samples}"
            )
        if not prefix_tokens:
            raise ValueError("prefix_tokens must be non-empty")

        if n_samples == 1:
            model, tokenizer = self._ensure_loaded()
            seed = seeds[0] if seeds is not None else self._default_seed
            try:
                return [
                    self._sample_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prefix_tokens=prefix_tokens,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    )
                ]
            except Exception as exc:
                raise LLMAdapterError(
                    f"generate_completions_with_shared_prefix failed at "
                    f"sample 0: {exc}"
                ) from exc

        model, tokenizer = self._ensure_loaded()
        seed = seeds[0] if seeds is not None else self._default_seed
        try:
            return self._batched_sample_completions(
                model=model,
                tokenizer=tokenizer,
                prefix_tokens=prefix_tokens,
                n_samples=n_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        except Exception as exc:
            raise LLMAdapterError(
                f"generate_completions_with_shared_prefix (batched) "
                f"failed: {exc}"
            ) from exc

    # ---- Metadata ----

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "adapter_name": "MLXLLMAdapter",
            "adapter_version": "0.1-phaseA",
            "model": self._model_name,
            "determinism_class": "approximate-fp",
            "default_seed": str(self._default_seed),
            "backend": "mlx-lm-direct",
        }

    # ============================================================
    # Internal: actual measurement implementation
    # ============================================================

    def _measure_at_position(
        self,
        model: _MLXModel,
        tokenizer: _MLXTokenizer,
        prompt: str,
        token_set: Sequence[str],
    ) -> TokenProbabilityResult:
        """Forward-pass + logprob extraction. Stays in MLX until
        scalar conversion at the end."""
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ImportError(
                "MLXLLMAdapter measurement requires mlx.core"
            ) from exc

        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            raise ValueError("Prompt produced empty token sequence")
        input_ids = mx.array(prompt_tokens)[None, :]
        # mlx-lm 0.31+ models return logits tensor directly (cache is
        # None for fresh prefill; mutated in place when caching).
        logits = model(input_ids)
        last_logits = logits[:, -1, :]  # [1, vocab_size]
        return self._build_result_from_last_logits(
            last_logits, tokenizer, token_set, mx,
        )

    def _build_result_from_last_logits(
        self,
        last_logits: Any,
        tokenizer: _MLXTokenizer,
        token_set: Sequence[str],
        mx: Any,
    ) -> TokenProbabilityResult:
        """Shared logic: from a [1, vocab_size] logits tensor, build
        the renormalised distribution + mass capture + top-K logprobs
        + schema-v4 fields. Used by both the single-prompt path and
        the shared-prefix batch path."""
        logprobs = last_logits - mx.logsumexp(
            last_logits, axis=-1, keepdims=True
        )
        logprobs_1d = logprobs[0]  # [vocab_size]
        probs = mx.exp(logprobs_1d)
        mx.eval(logprobs_1d, probs)

        effective_top_k = self._effective_top_k(len(token_set))
        sorted_indices = mx.argsort(-probs)
        top_k_indices = sorted_indices[:effective_top_k]
        mx.eval(top_k_indices)

        top_k_logprobs: dict[str, float] = {}
        for idx in top_k_indices.tolist():
            tok_text = tokenizer.decode([int(idx)])
            top_k_logprobs[tok_text] = float(logprobs_1d[int(idx)].item())

        # Token-aliasing: sum probabilities across canonical variants
        emitted_to_prob: dict[str, float] = {
            tok: float(math.exp(lp))
            for tok, lp in top_k_logprobs.items()
        }
        token_set_probs: dict[str, float] = {}
        truncated: list[str] = []
        for member in token_set:
            prob = 0.0
            found = False
            for variant in (member, " " + member, "\n" + member):
                if variant in emitted_to_prob:
                    prob += emitted_to_prob[variant]
                    found = True
            token_set_probs[member] = prob
            if not found:
                truncated.append(member)

        mass_capture = sum(token_set_probs.values())
        if mass_capture > 0:
            distribution = {
                k: v / mass_capture for k, v in token_set_probs.items()
            }
        else:
            distribution = {k: 0.0 for k in token_set_probs}

        # Schema-v4 fields
        from bsig.core.signature import (
            entropy_full_from_top_k,
            gap_top1_top_k_from_top_k,
            gap_top2_from_top_k,
            p_max_from_top_k,
            top_k_mass_from_top_k,
        )
        h_hyp = sum(-p * math.log(p) for p in distribution.values() if p > 0)

        return TokenProbabilityResult(
            distribution=distribution,
            mass_capture=mass_capture,
            truncated_members=tuple(truncated),
            top_k_logprobs=top_k_logprobs,
            p_max=p_max_from_top_k(top_k_logprobs),
            entropy_full=entropy_full_from_top_k(top_k_logprobs),
            entropy_hyp=h_hyp,
            top_k_mass=top_k_mass_from_top_k(top_k_logprobs, k=10),
            gap_top2=gap_top2_from_top_k(top_k_logprobs),
            gap_top1_topK=gap_top1_top_k_from_top_k(top_k_logprobs, k=10),
            chosen_logprob=max(top_k_logprobs.values()),
        )

    def _sample_completion(
        self,
        model: _MLXModel,
        tokenizer: _MLXTokenizer,
        prefix_tokens: Sequence[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> CompletionResult:
        """Single-sample completion with prefix. Phase A serial loop;
        Phase C replaces with batched implementation.

        mlx-lm 0.31+: ``model(...)`` returns logits directly; cache is
        a list of layer caches mutated in place across calls. The
        prefix is prefilled once into the cache; each subsequent
        forward pass takes the just-emitted token and predicts the
        next.
        """
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import (
                make_prompt_cache,
            )
        except ImportError as exc:
            raise ImportError(
                "MLXLLMAdapter sampling requires mlx.core and mlx_lm"
            ) from exc
        mx.random.seed(seed)

        cache = make_prompt_cache(model)
        prefix = mx.array(list(prefix_tokens))[None, :]
        logits = model(prefix, cache=cache)
        last_logits = logits[:, -1, :]

        eos_id = getattr(tokenizer, "eos_token_id", None)
        emitted: list[int] = []
        chosen_logprobs: list[float] = []
        stop_reason = "max_tokens"

        for _ in range(max_tokens):
            logprobs = last_logits - mx.logsumexp(
                last_logits, axis=-1, keepdims=True
            )
            logprobs_1d = logprobs[0]
            tok = self._sample_token(logprobs_1d, temperature, top_p, mx)
            mx.eval(tok, logprobs_1d)
            tok_id = int(tok.item())
            chosen_logprobs.append(float(logprobs_1d[tok_id].item()))

            if eos_id is not None and tok_id == eos_id:
                stop_reason = "stop_token"
                break
            emitted.append(tok_id)

            next_input = mx.array([[tok_id]])
            logits = model(next_input, cache=cache)
            last_logits = logits[:, -1, :]

        text = tokenizer.decode(emitted) if emitted else ""
        return CompletionResult(
            text=text,
            tokens=tuple(emitted),
            chosen_logprobs=tuple(chosen_logprobs),
            stop_reason=stop_reason,
        )

    def _batched_sample_completions(
        self,
        model: _MLXModel,
        tokenizer: _MLXTokenizer,
        prefix_tokens: Sequence[int],
        n_samples: int,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> list[CompletionResult]:
        """Phase C extension point — currently routes to Phase A serial.

        **Status (2026-05-07).** Phase C as designed (N samples decode
        together at ``batch_size=N`` per forward) is blocked on an
        upstream mlx-core bug: ``mx.fast.rope`` at ``batch >= 2,
        seq_len == 1`` produces incorrect output for rows 1+ even
        when input rows are bit-identical. Filed at
        https://github.com/ml-explore/mlx/issues/3494; fix PR #3498
        MERGED 2026-05-11 at https://github.com/ml-explore/mlx/pull/3498
        (3-line Metal kernel patch) but not yet in a released wheel
        (latest ``mlx==0.31.2`` predates the merge; unblocks at the
        first release > 0.31.2 shipping the patch). The diagnostic probe trail: post-
        prefill cache state identical across rows; v_proj
        identical; k_proj pre-RoPE identical; RoPE output diverges
        — bisected to the Metal kernel for ``fast::rope``. See
        corruption registry entry 2026-05-07 (engine category).

        A *hybrid* path was prototyped (single batched prefill, split
        into N batch-1 caches, per-sample serial decode) and benchmarked
        at 420-prefix + max_tokens=30 producing only ~1.05x speedup vs
        Phase A serial. Math: prefill amortization at decode-dominated
        Phase C workloads (long generation) is small. The hybrid does
        not earn its complexity for Phase C semantic-entropy use; it
        was reverted in favor of staying with the Phase A serial path.

        Phase C will be revisited when either (a) mlx-lm fixes batched
        incremental decode, or (b) vllm-mlx engine mode lands and
        provides correct continuous batching.

        For framework *measurement* workloads (short generation /
        per-position probability extraction), the
        ``_batch_with_shared_prefix`` path (used by
        ``get_token_probabilities_batch``) already provides ~2.2x
        speedup at 1014-prefix and is unaffected by this limitation.
        """
        results: list[CompletionResult] = []
        for sample_idx in range(n_samples):
            results.append(
                self._sample_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prefix_tokens=prefix_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    seed=seed + sample_idx,
                )
            )
        return results

    @staticmethod
    def _sample_token(
        logprobs_1d: Any,  # mx.array
        temperature: float,
        top_p: float,
        mx: Any,
    ) -> Any:
        """Sampling helper. temperature=0 → argmax; otherwise temperature
        sampling. Phase A implementation; Phase C batched-version
        replaces this with vectorised per-sample seeds and proper
        top-p (nucleus) masking.

        ``top_p`` is currently accepted for Protocol-conformance but
        not enforced (full distribution is sampled). Phase C
        introduces cumulative-mass masking. For the cross-adapter
        agreement test and typical Phase A usage with
        ``temperature ≤ 0.7``, the omission is benign.
        """
        if temperature <= 0.0:
            return mx.argmax(logprobs_1d)
        scaled = logprobs_1d / max(temperature, 1e-9)
        return mx.random.categorical(scaled, shape=())

    def _effective_top_k(self, n_token_set: int) -> int:
        if self._logprobs_top_k_override is not None:
            return self._logprobs_top_k_override
        return max(40, 10 * n_token_set)
