"""Tests for LlamaCppLLMAdapter via httpx.MockTransport.

The adapter speaks the OpenAI-compatible /v1/completions endpoint
exposed by llama.cpp's `llama-server`. These tests fake the wire
protocol with MockTransport — no real model, no real server. They
cover:

- Wire-protocol shape (request payload, response parsing).
- Token-aliasing across leading-space variants (" A" vs "A").
- Mass capture computed correctly from the raw distribution.
- Truncated-members handling: auto-extending retry once when a
  token_set member is below the heuristic top-K.
- Distribution renormalisation over token_set sums to 1.0.
- Per-call retry on transport failures (500s); per-item retry in
  batch with surgical-repair-friendly partial_results.
- Legacy hypothesis-distribution methods raise LLMAdapterError
  pointing at get_token_probabilities (per ADR-0008).
"""
from __future__ import annotations

import math

import pytest

httpx = pytest.importorskip("httpx")

from bsig.adapters.base import LLMAdapterError
from bsig.adapters.llm import TokenProbabilityResult
from bsig.reference.llm_llama_cpp import (
    LlamaCppLLMAdapter,
    _extract_token_probs,
)


# ---- Helpers ----


def _build_adapter(handler, **kwargs) -> LlamaCppLLMAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport, base_url="http://localhost:8080"
    )
    return LlamaCppLLMAdapter(_client=client, **kwargs)


def _completion_response(
    text: str,
    top_logprobs: list[dict] | None = None,
) -> httpx.Response:
    """Build a minimal /v1/completions response. If ``top_logprobs`` is
    not None, it appears at choices[0].logprobs.content[0].top_logprobs."""
    choice: dict = {"text": text, "index": 0}
    if top_logprobs is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "id": 0,
                    "token": top_logprobs[0]["token"] if top_logprobs else "",
                    "logprob": top_logprobs[0]["logprob"] if top_logprobs else 0.0,
                    "bytes": [],
                    "top_logprobs": top_logprobs,
                }
            ]
        }
    return httpx.Response(
        200,
        json={"choices": [choice]},
    )


def _logprobs_for(probs: dict[str, float]) -> list[dict]:
    """Build a top_logprobs list from {token: probability} mapping.
    Order follows insertion order; logprobs are computed from probs."""
    return [
        {"id": i, "token": tok, "logprob": math.log(p), "bytes": []}
        for i, (tok, p) in enumerate(probs.items())
        if p > 0
    ]


# ---- _extract_token_probs ----


def test_extract_simple_match() -> None:
    top = _logprobs_for({"A": 0.7, "B": 0.2, "C": 0.05, "D": 0.05})
    out = _extract_token_probs(top, ["A", "B", "C", "D"])
    assert out["A"] == pytest.approx(0.7, abs=1e-6)
    assert out["D"] == pytest.approx(0.05, abs=1e-6)


def test_extract_aliases_leading_space() -> None:
    top = _logprobs_for({" A": 0.6, " B": 0.4})
    out = _extract_token_probs(top, ["A", "B"])
    assert out["A"] == pytest.approx(0.6, abs=1e-6)
    assert out["B"] == pytest.approx(0.4, abs=1e-6)


def test_extract_sums_across_variants() -> None:
    """If both `A` and ` A` appear, sum them."""
    top = _logprobs_for({"A": 0.3, " A": 0.5, "B": 0.2})
    out = _extract_token_probs(top, ["A", "B"])
    assert out["A"] == pytest.approx(0.8, abs=1e-6)
    assert out["B"] == pytest.approx(0.2, abs=1e-6)


def test_extract_missing_member_absent_from_result() -> None:
    """If a token_set member has no variant in top_logprobs, it's not
    in the result dict (signals truncation upstream)."""
    top = _logprobs_for({" A": 0.6, " B": 0.4})  # no C, no D
    out = _extract_token_probs(top, ["A", "B", "C", "D"])
    assert "A" in out and "B" in out
    assert "C" not in out and "D" not in out


# ---- generate ----


def test_generate_returns_text() -> None:
    handler = lambda request: _completion_response("hello world")  # noqa: E731
    adapter = _build_adapter(handler)
    assert adapter.generate("any prompt") == "hello world"


def test_generate_retries_on_transient_failure() -> None:
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(500, json={"error": "transient"})
        return _completion_response("recovered")

    adapter = _build_adapter(handler)
    assert adapter.generate("any") == "recovered"
    assert state["calls"] == 2


def test_generate_exhausts_retries() -> None:
    handler = lambda request: httpx.Response(500, json={"error": "x"})  # noqa: E731
    adapter = _build_adapter(handler)
    with pytest.raises(LLMAdapterError, match="failed after"):
        adapter.generate("any", max_retries=1)


# ---- get_token_probabilities — happy path ----


def test_get_token_probabilities_distribution_sums_to_one() -> None:
    top = _logprobs_for({" A": 0.6, " B": 0.2, " C": 0.1, " D": 0.1})

    def handler(request):
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any prompt", ["A", "B", "C", "D"])
    assert isinstance(result, TokenProbabilityResult)
    total = sum(result.distribution.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_get_token_probabilities_populates_top_k_logprobs() -> None:
    """Schema-v3 (post-stage-4a-replication): TokenProbabilityResult
    carries the full top-K logprobs as raw measurement, including
    tokens NOT in token_set (markdown fences, alternative answers,
    etc.). Stored at full fidelity per the measurement-vs-computation
    methodology."""
    top = _logprobs_for({
        " A": 0.6, " B": 0.2, " C": 0.1, " D": 0.05, " ```": 0.04, " *": 0.01,
    })

    def handler(request):
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any", ["A", "B", "C", "D"])
    # All 6 tokens (including the non-token_set ones) are preserved
    assert len(result.top_k_logprobs) == 6
    assert " A" in result.top_k_logprobs
    assert " ```" in result.top_k_logprobs  # not in token_set; preserved anyway
    assert " *" in result.top_k_logprobs
    # Logprobs match the encoded probabilities
    assert math.exp(result.top_k_logprobs[" A"]) == pytest.approx(0.6, abs=1e-6)
    assert math.exp(result.top_k_logprobs[" ```"]) == pytest.approx(0.04, abs=1e-6)


def test_get_token_probabilities_top_k_reflects_extended_response_after_truncation() -> None:
    """When auto-extending retry fires (a token_set member missing from
    initial top-K), top_k_logprobs reflects the EXTENDED response, not
    the initial one — consistency with distribution + mass_capture
    which also use the extended response."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            # Initial: missing 'D'
            top = _logprobs_for({" A": 0.5, " B": 0.3, " C": 0.2})
        else:
            # Extended: includes 'D' plus extra non-letter tokens
            top = _logprobs_for(
                {" A": 0.5, " B": 0.3, " C": 0.15, " D": 0.04, " ```": 0.01}
            )
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any", ["A", "B", "C", "D"])
    assert state["calls"] == 2
    # top_k_logprobs comes from the extended response (5 tokens)
    assert len(result.top_k_logprobs) == 5
    assert " D" in result.top_k_logprobs
    assert " ```" in result.top_k_logprobs


def test_token_probability_result_default_top_k_logprobs_is_empty() -> None:
    """When TokenProbabilityResult is constructed without top_k_logprobs
    (e.g., in tests, by adapters that don't expose top-K), the field
    defaults to an empty mapping — *not* None — so consumers can call
    len() / iterate without guard."""
    r = TokenProbabilityResult(distribution={"A": 0.5, "B": 0.5}, mass_capture=1.0)
    assert r.top_k_logprobs == {}
    assert isinstance(r.top_k_logprobs, dict)


def test_get_token_probabilities_mass_capture_full() -> None:
    """When all token_set probability appears in top_logprobs, mass
    capture is the sum of those probabilities."""
    top = _logprobs_for({" A": 0.6, " B": 0.2, " C": 0.1, " D": 0.05})
    # Sum = 0.95 — leaving 5% on hypothetical other tokens

    def handler(request):
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any", ["A", "B", "C", "D"])
    assert result.mass_capture == pytest.approx(0.95, abs=1e-6)
    assert result.truncated_members == ()


def test_get_token_probabilities_renormalises_correctly() -> None:
    """After renormalisation, ranks are preserved and sum=1."""
    top = _logprobs_for({" A": 0.6, " B": 0.3, " ```": 0.05, " C": 0.04, " D": 0.01})

    def handler(request):
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any", ["A", "B", "C", "D"])
    # Mass = 0.6 + 0.3 + 0.04 + 0.01 = 0.95
    assert result.mass_capture == pytest.approx(0.95, abs=1e-6)
    # Rank preserved
    d = result.distribution
    assert d["A"] > d["B"] > d["C"] > d["D"]
    # A's renormalised probability = 0.6 / 0.95
    assert d["A"] == pytest.approx(0.6 / 0.95, abs=1e-6)


# ---- get_token_probabilities — truncation ----


def test_get_token_probabilities_truncation_retries_then_fills_zero() -> None:
    """When a token_set member is missing from top_logprobs, the
    adapter retries with extended top_k. If still missing, it
    populates truncated_members and returns P=0 for that member."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        # First call: small top-K missing 'D'
        # Second call (retry with 4×): also missing 'D' → truncated
        top = _logprobs_for({" A": 0.5, " B": 0.3, " C": 0.2})
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any", ["A", "B", "C", "D"])
    assert state["calls"] == 2  # heuristic + one auto-extending retry
    assert result.truncated_members == ("D",)
    assert result.distribution["D"] == 0.0
    # A/B/C renormalised over their captured mass
    assert result.distribution["A"] == pytest.approx(0.5 / 1.0, abs=1e-6)


def test_get_token_probabilities_truncation_resolved_on_retry() -> None:
    """If the auto-extending retry recovers the missing member,
    truncated_members stays empty."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            top = _logprobs_for({" A": 0.5, " B": 0.3, " C": 0.2})
        else:
            top = _logprobs_for(
                {" A": 0.5, " B": 0.3, " C": 0.15, " D": 0.05}
            )
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    result = adapter.get_token_probabilities("any", ["A", "B", "C", "D"])
    assert state["calls"] == 2
    assert result.truncated_members == ()
    assert result.distribution["D"] > 0
    assert result.mass_capture == pytest.approx(1.0, abs=1e-6)


def test_get_token_probabilities_zero_mass_raises() -> None:
    """If no token_set member appears in top_logprobs even after retry,
    mass=0 and the renormalisation is undefined → raise."""
    top = _logprobs_for({" something_else": 0.6, " other": 0.4})

    def handler(request):
        return _completion_response(text=" something_else", top_logprobs=top)

    adapter = _build_adapter(handler)
    with pytest.raises(LLMAdapterError, match="zero probability"):
        adapter.get_token_probabilities("any", ["A", "B", "C", "D"])


# ---- get_token_probabilities — request shape ----


def test_get_token_probabilities_sends_logprobs_request() -> None:
    """The adapter requests logprobs from the server."""
    captured = {}

    def handler(request):
        captured["payload"] = request.read()
        top = _logprobs_for({" A": 0.6, " B": 0.4})
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    adapter.get_token_probabilities("test prompt", ["A", "B"])
    import json as _json
    payload = _json.loads(captured["payload"])
    assert "logprobs" in payload
    assert payload["logprobs"] >= 40  # default heuristic
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 1
    assert payload["prompt"] == "test prompt"


def test_get_token_probabilities_uses_constructor_logprobs_top_k() -> None:
    captured = {}

    def handler(request):
        captured["payload"] = request.read()
        top = _logprobs_for({" A": 0.6, " B": 0.4})
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler, logprobs_top_k=99)
    adapter.get_token_probabilities("any", ["A", "B"])
    import json as _json
    payload = _json.loads(captured["payload"])
    assert payload["logprobs"] == 99


def test_get_token_probabilities_passes_grammar_when_set() -> None:
    captured = {}

    def handler(request):
        captured["payload"] = request.read()
        top = _logprobs_for({" A": 0.6, " B": 0.4})
        return _completion_response(text=" A", top_logprobs=top)

    grammar = 'root ::= "A" | "B"'
    adapter = _build_adapter(handler, output_grammar=grammar)
    adapter.get_token_probabilities("any", ["A", "B"])
    import json as _json
    payload = _json.loads(captured["payload"])
    assert payload["grammar"] == grammar


# ---- get_token_probabilities_batch ----


def test_get_token_probabilities_batch_returns_per_prompt() -> None:
    top = _logprobs_for({" A": 0.6, " B": 0.4})

    def handler(request):
        return _completion_response(text=" A", top_logprobs=top)

    adapter = _build_adapter(handler)
    results = adapter.get_token_probabilities_batch(
        ["p1", "p2", "p3"], ["A", "B"]
    )
    assert len(results) == 3
    assert all(isinstance(r, TokenProbabilityResult) for r in results)


def test_get_token_probabilities_batch_failure_provides_partial_results() -> None:
    """When item k fails, the raised LLMAdapterError carries the
    successful items 0..k-1 and Nones for k onward (per ADR-0008
    permissive-but-encouraged surgical-repair contract)."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        # Items 0 and 1 succeed; item 2 fails permanently
        # Each item costs 1 call (no truncation retry, all letters present)
        if state["calls"] <= 2:
            top = _logprobs_for({" A": 0.6, " B": 0.4})
            return _completion_response(text=" A", top_logprobs=top)
        return httpx.Response(500, json={"error": "x"})

    adapter = _build_adapter(handler)
    with pytest.raises(LLMAdapterError) as excinfo:
        adapter.get_token_probabilities_batch(
            ["p1", "p2", "p3"], ["A", "B"], max_retries=1
        )
    assert excinfo.value.failed_index == 2
    partial = excinfo.value.partial_results
    assert isinstance(partial, list)
    assert len(partial) == 3
    assert partial[0] is not None and partial[1] is not None
    assert partial[2] is None


# ---- get_metadata ----


def test_metadata_has_required_fields() -> None:
    handler = lambda request: _completion_response("x")  # noqa: E731
    adapter = _build_adapter(handler, model="qwen2.5:7b-instruct")
    md = adapter.get_metadata()
    assert md["adapter_name"] == "LlamaCppLLMAdapter"
    assert md["model"] == "qwen2.5:7b-instruct"
    assert md["determinism_class"] == "approximate-fp"


def test_metadata_records_logprobs_top_k_override() -> None:
    handler = lambda request: _completion_response("x")  # noqa: E731
    adapter = _build_adapter(handler, logprobs_top_k=99)
    assert adapter.get_metadata()["logprobs_top_k_override"] == "99"


def test_metadata_default_logprobs_top_k_is_auto() -> None:
    handler = lambda request: _completion_response("x")  # noqa: E731
    adapter = _build_adapter(handler)
    assert adapter.get_metadata()["logprobs_top_k_override"] == "auto"


# ---- Effective top-K heuristic ----


def test_effective_top_k_uses_heuristic_for_4_letters() -> None:
    handler = lambda request: _completion_response("x")  # noqa: E731
    adapter = _build_adapter(handler)
    # max(40, 10 × 4) = 40
    assert adapter._effective_top_k(4) == 40


def test_effective_top_k_scales_with_token_set_size() -> None:
    handler = lambda request: _completion_response("x")  # noqa: E731
    adapter = _build_adapter(handler)
    # max(40, 10 × 25) = 250
    assert adapter._effective_top_k(25) == 250


def test_effective_top_k_constructor_override() -> None:
    handler = lambda request: _completion_response("x")  # noqa: E731
    adapter = _build_adapter(handler, logprobs_top_k=99)
    assert adapter._effective_top_k(4) == 99
    assert adapter._effective_top_k(25) == 99
