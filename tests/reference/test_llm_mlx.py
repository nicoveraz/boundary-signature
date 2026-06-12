"""Tests for MLXLLMAdapter Phase A scaffolding.

The adapter is tested via an injected fake loader that returns a fake
model + fake tokenizer. Real MLX is not a test dependency; the cross-
adapter agreement test (§7.1 of stage_6_mlx_adapter pre-design) is a
separate hardware-only test that exercises the actual mlx-lm path.

Phase A unit-test scope:
- Constructor + lazy-load semantics
- Protocol-conformance shape (methods exist, return correct types)
- Mass-capture / renormalisation logic against deterministic fake
  logits (the ALGORITHMIC core of the adapter, separable from MLX)
- Metadata reporting

NOT in scope here (separate hardware-only tests):
- Real mlx-lm integration
- Numerical agreement vs llama.cpp
- Phase C batched generation correctness (Phase A serial loop is
  scaffolding only).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import math
import pytest

from bsig.adapters.base import LLMAdapterError
from bsig.adapters.llm import CompletionResult, TokenProbabilityResult
from bsig.reference.llm_mlx import MLXLLMAdapter


# ============================================================
# Fake mlx-lm shapes for testing without MLX installed
# ============================================================


class _FakeTokenizer:
    """Deterministic tokenizer: each character maps to its ord; decode
    inverts. Sufficient for testing top-K extraction + renormalisation
    logic; not a realistic tokenizer."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [ord(c) for c in text]

    def decode(self, tokens) -> str:
        return "".join(chr(int(t)) for t in tokens if int(t) != 0)


class _FakeModel:
    """Fake model that returns caller-supplied logits per call.

    Since the adapter calls ``mx.exp`` and ``mx.logsumexp`` on the
    returned tensors, we'd need real MLX to actually run forward
    passes. Phase A unit tests therefore use a higher-level mock —
    the entire ``_measure_at_position`` is mocked, and we verify the
    Protocol shape of the adapter.

    For deeper algorithmic tests of mass capture / renormalisation,
    see ``test_measure_at_position_logic`` which exercises the
    algorithm directly against numpy without mlx.
    """

    def __init__(self, return_value: Any = None) -> None:
        self.return_value = return_value or (None, None)
        self.call_count = 0

    def __call__(self, input_ids, cache=None):
        self.call_count += 1
        return self.return_value


def _fake_loader(model_name: str) -> tuple[Any, Any]:
    return _FakeModel(), _FakeTokenizer()


# ============================================================
# Constructor / lazy load
# ============================================================


def test_constructor_does_not_load_eagerly() -> None:
    """Loader factory should not be called at construction (lazy)."""
    calls: list[str] = []

    def tracking_loader(name: str):
        calls.append(name)
        return _FakeModel(), _FakeTokenizer()

    adapter = MLXLLMAdapter(model="test", _loader_factory=tracking_loader)
    assert calls == []  # lazy: nothing loaded yet


def test_loader_factory_called_on_first_use(monkeypatch) -> None:
    calls: list[str] = []

    def tracking_loader(name: str):
        calls.append(name)
        return _FakeModel(), _FakeTokenizer()

    adapter = MLXLLMAdapter(
        model="test-model", _loader_factory=tracking_loader
    )
    # _ensure_loaded triggers the factory
    adapter._ensure_loaded()
    assert calls == ["test-model"]
    # Subsequent call doesn't re-load
    adapter._ensure_loaded()
    assert calls == ["test-model"]


def test_loader_failure_raises_llm_adapter_error() -> None:
    def failing_loader(name: str):
        raise RuntimeError("simulated load failure")

    adapter = MLXLLMAdapter(_loader_factory=failing_loader)
    with pytest.raises(LLMAdapterError, match="failed to load"):
        adapter._ensure_loaded()


# ============================================================
# Protocol shape
# ============================================================


def test_satisfies_llm_adapter_protocol_methods() -> None:
    """Structural typing check: adapter exposes all required methods."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    for method in (
        "generate",
        "generate_batch",
        "get_token_probabilities",
        "get_token_probabilities_batch",
        "get_metadata",
        "generate_completions_with_shared_prefix",
    ):
        assert callable(getattr(adapter, method))


def test_get_metadata_includes_required_fields() -> None:
    adapter = MLXLLMAdapter(model="qwen-test", _loader_factory=_fake_loader)
    md = adapter.get_metadata()
    assert md["adapter_name"] == "MLXLLMAdapter"
    assert md["model"] == "qwen-test"
    assert "adapter_version" in md
    assert "determinism_class" in md
    assert "backend" in md


def test_metadata_reports_phase() -> None:
    """Metadata should make the implementation phase explicit (not the
    certified production adapter). Phase-A through Phase-C versions
    are the legitimate values prior to vllm-mlx engine mode."""
    md = MLXLLMAdapter(_loader_factory=_fake_loader).get_metadata()
    version = md["adapter_version"].lower()
    assert "phase" in version
    assert any(stage in version for stage in ("phasea", "phaseb", "phasec"))


# ============================================================
# Algorithmic tests of measurement logic
# ============================================================
#
# These exercise the mass-capture / renormalisation algorithm
# directly via a mock _measure_at_position to avoid needing real MLX
# tensors. The actual MLX-tensor path is exercised by the cross-
# adapter agreement test on hardware.


def test_get_token_probabilities_returns_protocol_shape(monkeypatch) -> None:
    """If _measure_at_position returns a TokenProbabilityResult,
    get_token_probabilities passes it through."""
    fake_result = TokenProbabilityResult(
        distribution={"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
        mass_capture=0.85,
        truncated_members=(),
        top_k_logprobs={"A": -0.92, "B": -1.20, "C": -1.61, "D": -2.30},
    )

    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    monkeypatch.setattr(
        adapter,
        "_measure_at_position",
        lambda *args, **kwargs: fake_result,
    )
    result = adapter.get_token_probabilities("test prompt", ["A", "B", "C", "D"])
    assert result is fake_result
    assert isinstance(result, TokenProbabilityResult)
    assert math.isclose(sum(result.distribution.values()), 1.0)
    assert 0.0 <= result.mass_capture <= 1.0


def test_get_token_probabilities_wraps_exceptions_as_llm_adapter_error(monkeypatch) -> None:
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated MLX failure")

    monkeypatch.setattr(adapter, "_measure_at_position", boom)
    with pytest.raises(LLMAdapterError, match="get_token_probabilities failed"):
        adapter.get_token_probabilities("p", ["A", "B"])


def test_get_token_probabilities_batch_per_item_partial_results(monkeypatch) -> None:
    """Surgical-repair compliance (parallels llama.cpp adapter test):
    batch failure carries partial_results."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)

    call_count = {"n": 0}

    def fake_measure(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return TokenProbabilityResult(
                distribution={"A": 1.0, "B": 0.0},
                mass_capture=0.99,
                top_k_logprobs={"A": -0.01, "B": -10.0},
            )
        raise RuntimeError("simulated failure on second call")

    monkeypatch.setattr(adapter, "_measure_at_position", fake_measure)
    with pytest.raises(LLMAdapterError) as excinfo:
        adapter.get_token_probabilities_batch(["p1", "p2"], ["A", "B"])
    exc = excinfo.value
    assert exc.failed_index == 1
    assert exc.partial_results is not None
    partial = exc.partial_results
    assert isinstance(partial, list)
    assert isinstance(partial[0], TokenProbabilityResult)
    assert partial[1] is None


# ============================================================
# Effective top-K heuristic
# ============================================================


def test_effective_top_k_default_heuristic() -> None:
    """max(40, 10 × len(token_set)) per spec."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    assert adapter._effective_top_k(4) == 40   # max(40, 40) = 40
    assert adapter._effective_top_k(10) == 100  # max(40, 100) = 100
    assert adapter._effective_top_k(2) == 40   # max(40, 20) = 40


def test_effective_top_k_constructor_override() -> None:
    adapter = MLXLLMAdapter(
        logprobs_top_k=80, _loader_factory=_fake_loader,
    )
    assert adapter._effective_top_k(4) == 80
    assert adapter._effective_top_k(10) == 80


# ============================================================
# generate_completions_with_shared_prefix Protocol shape
# ============================================================


def test_generate_completions_with_shared_prefix_validates_seeds_length() -> None:
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    with pytest.raises(ValueError, match="len.seeds"):
        adapter.generate_completions_with_shared_prefix(
            prefix_tokens=[1, 2, 3],
            n_samples=5,
            max_tokens=10,
            temperature=0.7,
            seeds=[1, 2],  # wrong length
        )


def test_generate_completions_with_shared_prefix_n1_uses_serial_path(
    monkeypatch,
) -> None:
    """n_samples=1 short-circuits to the serial (Phase A) path, calling
    _sample_completion once with seeds[0] (or default_seed if None)."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    captured_seeds: list[int] = []

    def fake_sample(self, model, tokenizer, prefix_tokens, max_tokens,
                    temperature, top_p, seed):
        captured_seeds.append(seed)
        return CompletionResult(
            text=f"sample_seed_{seed}",
            tokens=(seed,),
            chosen_logprobs=(-0.5,),
            stop_reason="max_tokens",
        )

    monkeypatch.setattr(MLXLLMAdapter, "_sample_completion", fake_sample)
    completions = adapter.generate_completions_with_shared_prefix(
        prefix_tokens=[10, 20, 30],
        n_samples=1,
        max_tokens=5,
        temperature=0.7,
        seeds=[100],
    )
    assert len(completions) == 1
    assert completions[0].text == "sample_seed_100"
    assert captured_seeds == [100]


def test_generate_completions_n1_uses_default_seed_when_none(
    monkeypatch,
) -> None:
    """When n_samples=1 and seeds=None, the adapter's default_seed is
    forwarded to _sample_completion."""
    adapter = MLXLLMAdapter(default_seed=42, _loader_factory=_fake_loader)
    captured_seeds: list[int] = []

    def fake_sample(self, model, tokenizer, prefix_tokens, max_tokens,
                    temperature, top_p, seed):
        captured_seeds.append(seed)
        return CompletionResult(
            text=f"s{seed}", tokens=(0,), chosen_logprobs=(-0.1,),
        )

    monkeypatch.setattr(MLXLLMAdapter, "_sample_completion", fake_sample)
    adapter.generate_completions_with_shared_prefix(
        prefix_tokens=[1], n_samples=1, max_tokens=2, temperature=0.5,
    )
    assert captured_seeds == [42]


def test_generate_completions_n_gt_1_uses_batched_path(monkeypatch) -> None:
    """n_samples > 1 routes to _batched_sample_completions with seeds[0]
    as the global seed; _sample_completion is NOT called."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    captured: dict[str, object] = {}

    def fake_batched(
        self, model, tokenizer, prefix_tokens, n_samples, max_tokens,
        temperature, seed,
    ):
        captured["n_samples"] = n_samples
        captured["seed"] = seed
        captured["temperature"] = temperature
        return [
            CompletionResult(
                text=f"batch_{i}", tokens=(i,), chosen_logprobs=(-0.1,),
                stop_reason="max_tokens",
            )
            for i in range(n_samples)
        ]

    def fake_sample(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError(
            "n_samples > 1 must NOT call _sample_completion"
        )

    monkeypatch.setattr(
        MLXLLMAdapter, "_batched_sample_completions", fake_batched
    )
    monkeypatch.setattr(MLXLLMAdapter, "_sample_completion", fake_sample)

    completions = adapter.generate_completions_with_shared_prefix(
        prefix_tokens=[10, 20, 30],
        n_samples=3,
        max_tokens=5,
        temperature=0.7,
        seeds=[100, 200, 300],
    )
    assert len(completions) == 3
    assert [c.text for c in completions] == ["batch_0", "batch_1", "batch_2"]
    assert captured["n_samples"] == 3
    assert captured["seed"] == 100  # seeds[0] used as global
    assert captured["temperature"] == 0.7


def test_generate_completions_n_gt_1_default_seed_when_none(
    monkeypatch,
) -> None:
    """Without explicit seeds, the adapter's default_seed is the global
    rng seed for the batched path."""
    adapter = MLXLLMAdapter(default_seed=99, _loader_factory=_fake_loader)
    captured_seed: list[int] = []

    def fake_batched(
        self, model, tokenizer, prefix_tokens, n_samples, max_tokens,
        temperature, seed,
    ):
        captured_seed.append(seed)
        return [
            CompletionResult(
                text="x", tokens=(0,), chosen_logprobs=(-0.1,),
                stop_reason="max_tokens",
            )
            for _ in range(n_samples)
        ]

    monkeypatch.setattr(
        MLXLLMAdapter, "_batched_sample_completions", fake_batched
    )
    adapter.generate_completions_with_shared_prefix(
        prefix_tokens=[1], n_samples=3, max_tokens=2, temperature=0.5,
    )
    assert captured_seed == [99]


def test_generate_completions_rejects_empty_prefix() -> None:
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    with pytest.raises(ValueError, match="non-empty"):
        adapter.generate_completions_with_shared_prefix(
            prefix_tokens=[],
            n_samples=2,
            max_tokens=4,
            temperature=0.5,
        )


def test_generate_completions_rejects_zero_samples() -> None:
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    with pytest.raises(ValueError, match="n_samples"):
        adapter.generate_completions_with_shared_prefix(
            prefix_tokens=[1, 2, 3],
            n_samples=0,
            max_tokens=4,
            temperature=0.5,
        )


# ============================================================
# generate_batch surgical-repair compliance
# ============================================================


def test_generate_batch_failure_carries_partial_results(monkeypatch) -> None:
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)

    call_count = {"n": 0}

    def fake_generate(self, prompt, max_tokens=None, temperature=0.0, max_retries=2):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return f"result-{call_count['n']}"
        raise LLMAdapterError("simulated MLX-side failure")

    monkeypatch.setattr(MLXLLMAdapter, "generate", fake_generate)
    with pytest.raises(LLMAdapterError) as excinfo:
        adapter.generate_batch(["p1", "p2"])
    exc = excinfo.value
    assert exc.failed_index == 1
    assert exc.partial_results is not None
    partial = exc.partial_results
    assert isinstance(partial, list)
    assert partial[0] == "result-1"
    assert partial[1] is None


# ---- Phase A polish: shared-prefix cache reuse in batch ----


def test_longest_common_prefix_len_full_match() -> None:
    """All identical sequences → LCP is full length."""
    sequences = [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert MLXLLMAdapter._longest_common_prefix_len(sequences) == 4


def test_longest_common_prefix_len_partial() -> None:
    """Shared prefix of length 3, divergence at position 3."""
    sequences = [[1, 2, 3, 4, 5], [1, 2, 3, 9, 9], [1, 2, 3, 7, 7]]
    assert MLXLLMAdapter._longest_common_prefix_len(sequences) == 3


def test_longest_common_prefix_len_no_shared() -> None:
    """No shared prefix → 0."""
    sequences = [[1, 2, 3], [4, 5, 6]]
    assert MLXLLMAdapter._longest_common_prefix_len(sequences) == 0


def test_longest_common_prefix_len_empty_input() -> None:
    assert MLXLLMAdapter._longest_common_prefix_len([]) == 0


def test_longest_common_prefix_len_handles_different_lengths() -> None:
    """LCP is bounded by the shortest sequence."""
    sequences = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    assert MLXLLMAdapter._longest_common_prefix_len(sequences) == 2


def test_batch_falls_back_to_sequential_on_no_shared_prefix(monkeypatch) -> None:
    """When prompts have no substantial common prefix, batch path
    falls back to per-prompt fresh forward (equivalent to looping
    over get_token_probabilities)."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)

    # Mock get_token_probabilities to return distinguishable results
    call_count = {"n": 0}
    def fake_single(self, prompt, token_set, max_retries=2):
        call_count["n"] += 1
        return TokenProbabilityResult(
            distribution={"A": 1.0, "B": 0.0},
            mass_capture=0.99,
            top_k_logprobs={"A": -0.01},
        )
    monkeypatch.setattr(MLXLLMAdapter, "get_token_probabilities", fake_single)

    # Three prompts with NO shared prefix
    prompts = ["alpha", "beta", "gamma"]
    results = adapter.get_token_probabilities_batch(prompts, ["A", "B"])
    assert len(results) == 3
    assert call_count["n"] == 3  # sequential fallback used


def test_batch_single_prompt_uses_get_token_probabilities(monkeypatch) -> None:
    """Single-prompt batch shortcuts to get_token_probabilities (no
    caching overhead on a degenerate batch)."""
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)

    call_count = {"n": 0}
    def fake_single(self, prompt, token_set, max_retries=2):
        call_count["n"] += 1
        return TokenProbabilityResult(
            distribution={"A": 1.0, "B": 0.0},
            mass_capture=0.99,
            top_k_logprobs={"A": -0.01},
        )
    monkeypatch.setattr(MLXLLMAdapter, "get_token_probabilities", fake_single)

    results = adapter.get_token_probabilities_batch(["only"], ["A", "B"])
    assert len(results) == 1
    assert call_count["n"] == 1


def test_batch_empty_input_returns_empty_list() -> None:
    adapter = MLXLLMAdapter(_loader_factory=_fake_loader)
    assert list(adapter.get_token_probabilities_batch([], ["A", "B"])) == []
