"""Reference LLMAdapter for Ollama-served models.

Lazy-imports ``httpx`` (in the ``ollama`` extra). Default endpoint is
``http://localhost:11434``. Default model is ``qwen2.5:7b-instruct``
(per CLAUDE.md §8 + the stage-3 dress-rehearsal exploration).

Implements the text-generation methods (``generate``,
``generate_batch``) and metadata. Does **not** implement
``get_token_probabilities`` — Ollama's API does not expose next-token
logprobs in a usable form; consumers needing token-probability
measurement should use :class:`bsig.reference.llm_llama_cpp.
LlamaCppLLMAdapter` instead. This adapter is retained for the smoke-
test path and as a baseline transport for text generation.

Constructor is **pure**: no pre-flight HTTP call, no Ollama-
availability check. Failures surface at first ``generate`` call with
a clear hint (``ollama pull <model>``) when the model is missing.

A single ``httpx.Client`` is constructed lazily on first call and
reused for the adapter's lifetime; cleanup via ``__del__``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from bsig.adapters.base import LLMAdapterError

if TYPE_CHECKING:
    import httpx


_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaLLMAdapter:
    """LLMAdapter implementation backed by Ollama.

    Constructor parameters:
    - ``model``: Ollama model name (must be pulled; verified at first
      call, not at construction).
    - ``host``: Ollama API base URL.
    - ``timeout``: per-request HTTP timeout in seconds.
    - ``default_seed``: ``options.seed`` value sent on every call (per
      Ollama API). Determinism aid; combined with ``temperature=0.0``
      this produces reproducible outputs.
    - ``_client``: optional pre-built ``httpx.Client`` for tests
      (typically with ``MockTransport``). Default None means construct
      a standard client lazily.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        timeout: float = 180.0,
        default_seed: int = 42,
        _client: "httpx.Client | None" = None,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout = timeout
        self._default_seed = default_seed
        self._client = _client

    def __del__(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # ---- Single-item methods ----

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
                return self._call_ollama(prompt, temperature, max_tokens)
            except _ModelNotFoundError as exc:
                # Model missing — no retry. Re-raise as LLMAdapterError
                # with the actionable hint per stage-3.5a OQ4 lazy
                # verification.
                raise LLMAdapterError(
                    f"Ollama model {self._model!r} not found. "
                    f"Run: `ollama pull {self._model}`"
                ) from exc
            except Exception as exc:
                last_exc = exc
        raise LLMAdapterError(
            f"OllamaLLMAdapter.generate failed after "
            f"{max_retries + 1} attempts (model={self._model!r})"
        ) from last_exc

    # ---- Batch methods (loops with surgical-repair compliance) ----

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

    # ---- Metadata ----

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "adapter_name": "OllamaLLMAdapter",
            "adapter_version": "1",
            "model": self._model,
            "host": self._host,
        }

    # ---- Internal HTTP plumbing ----

    def _get_client(self) -> "httpx.Client":
        if self._client is None:
            try:
                import httpx  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "OllamaLLMAdapter requires httpx. Install with: "
                    "uv pip install -e '.[ollama]'"
                ) from exc
            self._client = httpx.Client(
                timeout=self._timeout, base_url=self._host
            )
        return self._client

    def _call_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Single HTTP call to Ollama. No retries — caller wraps."""
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "OllamaLLMAdapter requires httpx. Install with: "
                "uv pip install -e '.[ollama]'"
            ) from exc

        client = self._get_client()
        options: dict[str, Any] = {
            "temperature": temperature,
            "seed": self._default_seed,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        try:
            response = client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Detect model-not-found 404 for the actionable hint
            # (stage-3.5a OQ4 lazy verification).
            if exc.response.status_code == 404:
                text = exc.response.text.lower()
                if "model" in text or "not found" in text:
                    raise _ModelNotFoundError(self._model) from exc
            raise

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Ollama returned non-JSON response: {response.text[:200]}"
            ) from exc

        if "response" not in data:
            raise KeyError(
                f"Ollama response missing 'response' key: {list(data)}"
            )
        return str(data["response"])


# ---- Internal sentinel exception (caught and re-raised by callers) ----


class _ModelNotFoundError(Exception):
    """Internal: raised by ``_call_ollama`` when Ollama returns 404
    with model-not-found body. Caller catches and re-raises as
    ``LLMAdapterError`` with the actionable hint."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Ollama model {model!r} not found")
        self.model = model
