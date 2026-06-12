"""Tests for OllamaLLMAdapter via httpx.MockTransport."""
from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")

from bsig.adapters.base import LLMAdapterError
from bsig.reference.llm_local import OllamaLLMAdapter


# ---- Helpers ----


def _build_adapter(handler, model: str = "test-model") -> OllamaLLMAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport, base_url="http://localhost:11434"
    )
    return OllamaLLMAdapter(model=model, _client=client)


def _ok_handler(response_text: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": response_text})
    return handler


def _failing_then_ok_handler(response_text: str, n_failures: int = 1):
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] <= n_failures:
            return httpx.Response(500, json={"error": "transient"})
        return httpx.Response(200, json={"response": response_text})

    return handler, state


def _model_not_found_handler(model_name: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": f"model '{model_name}' not found, "
                f"try pulling it first"
            },
        )
    return handler


# ---- generate ----


def test_generate_success() -> None:
    adapter = _build_adapter(_ok_handler("hello world"))
    assert adapter.generate("any prompt") == "hello world"


def test_generate_retries_on_5xx_then_succeeds() -> None:
    handler, state = _failing_then_ok_handler("recovered", n_failures=1)
    adapter = _build_adapter(handler)
    result = adapter.generate("prompt", max_retries=2)
    assert result == "recovered"
    assert state["calls"] == 2


def test_generate_exhausts_retries_then_raises() -> None:
    """All attempts return 500 → LLMAdapterError after max_retries+1."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(500, json={"error": "permanent"})

    adapter = _build_adapter(handler)
    with pytest.raises(LLMAdapterError, match="failed after"):
        adapter.generate("prompt", max_retries=2)
    assert state["calls"] == 3  # initial + 2 retries


def test_generate_model_not_found_includes_pull_hint() -> None:
    """Per stage-3.5a OQ4: model-not-found errors carry actionable hint."""
    adapter = _build_adapter(_model_not_found_handler("missing-model"), model="missing-model")
    with pytest.raises(LLMAdapterError) as excinfo:
        adapter.generate("prompt")
    msg = str(excinfo.value)
    assert "missing-model" in msg
    assert "ollama pull" in msg


def test_generate_model_not_found_does_not_retry() -> None:
    """404 model-not-found is non-transient; should not consume retries."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(
            404, json={"error": "model not found"}
        )

    adapter = _build_adapter(handler)
    with pytest.raises(LLMAdapterError):
        adapter.generate("prompt", max_retries=5)
    assert state["calls"] == 1  # no retries


# ---- generate_batch (surgical-repair compliance) ----


def test_generate_batch_all_success() -> None:
    adapter = _build_adapter(_ok_handler("response"))
    results = adapter.generate_batch(["p1", "p2", "p3"])
    assert results == ["response", "response", "response"]


def test_generate_batch_failure_provides_partial_results() -> None:
    """Per S5_2: on batch failure, partial_results carries successful
    items at non-failed positions."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        # First two calls succeed; third (and all retries) fail
        if state["calls"] <= 2:
            return httpx.Response(200, json={"response": f"r{state['calls']}"})
        return httpx.Response(500, json={"error": "permanent"})

    adapter = _build_adapter(handler)
    with pytest.raises(LLMAdapterError) as excinfo:
        adapter.generate_batch(["p1", "p2", "p3", "p4"], max_retries=1)

    exc = excinfo.value
    assert exc.failed_index == 2  # third item (0-indexed)
    assert exc.partial_results is not None
    partial = exc.partial_results
    assert isinstance(partial, list)
    assert partial[0] == "r1"
    assert partial[1] == "r2"
    assert partial[2] is None  # failed position
    assert partial[3] is None  # not-yet-attempted


# ---- Metadata ----


def test_get_metadata_includes_model_and_host() -> None:
    adapter = _build_adapter(_ok_handler(""), model="test-model")
    md = adapter.get_metadata()
    assert md["adapter_name"] == "OllamaLLMAdapter"
    assert md["model"] == "test-model"
    assert "host" in md


# ---- Protocol satisfaction ----


def test_satisfies_llm_adapter_protocol() -> None:
    """Structural typing check."""
    from bsig.adapters.llm import LLMAdapter

    adapter: LLMAdapter = _build_adapter(_ok_handler("hello"))
    # All five methods callable
    assert adapter.generate("p") == "hello"
    assert adapter.get_metadata()["adapter_name"] == "OllamaLLMAdapter"
