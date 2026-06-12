"""Shared fixtures for MedQA tests.

- ``deterministic_mock_embedder``: hashes text to seed an RNG, produces
  deterministic L2-normalized embeddings. Use for canonicalizer tests.
- ``ScriptedMockLLM``: configurable mock LLM with caller-supplied
  ``generate_fn`` and ``token_probabilities_fn``.
- ``FixedResponseLLM``: returns a fixed string and uniform token
  probabilities; useful when the test only cares about response shape.

LLM mocks are stage-3.3 work but conftest is the right place to hold
them as fixtures emerge. Stage 3.1 only needs the embedder mock.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from bsig.adapters.llm import TokenProbabilityResult


class DeterministicMockEmbedder:
    """L2-normalized embeddings derived from a hash of the input text.

    Same text → same embedding across runs. Sufficient for canonicalizer
    correctness tests; not realistic enough for actual signature
    computation (which needs semantic similarity).
    """

    def __init__(self, dim: int = 8) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts])

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "model": "DeterministicMockEmbedder",
            "model_version": "1",
            "dim": str(self._dim),
        }

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def mock_embedder() -> DeterministicMockEmbedder:
    return DeterministicMockEmbedder(dim=8)


# ---- Mock LLMs ----


class ScriptedMockLLM:
    """LLM mock with caller-supplied response functions.

    Default behavior: ``generate`` returns a canonical "3-step CoT
    ending in 'Final answer: A'" string for any prompt;
    ``get_token_probabilities`` (per ADR-0008) returns uniform with
    full mass capture and no truncation.

    Override via ``generate_fn`` and ``token_probabilities_fn``
    callables for test-specific scenarios.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        token_probabilities_fn: Callable[
            [str, Sequence[str]], "TokenProbabilityResult"
        ] | None = None,
    ) -> None:
        self._generate_fn = generate_fn or _default_generate
        self._token_probabilities_fn = (
            token_probabilities_fn or _default_token_probabilities
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> str:
        return self._generate_fn(prompt)

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Sequence[str]:
        return [self._generate_fn(p) for p in prompts]

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> "TokenProbabilityResult":
        return self._token_probabilities_fn(prompt, token_set)

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> Sequence["TokenProbabilityResult"]:
        return [self._token_probabilities_fn(p, token_set) for p in prompts]

    def get_metadata(self) -> Mapping[str, str]:
        return {"model": "ScriptedMockLLM", "model_version": "1"}


def _default_generate(prompt: str) -> str:
    return (
        "Reasoning step 1: Consider the patient's symptoms.\n"
        "Reasoning step 2: The lab values point toward A.\n"
        "Reasoning step 3: A is most consistent.\n"
        "\n"
        "Final answer: A\n"
    )


def _default_token_probabilities(
    prompt: str, token_set: Sequence[str]
) -> "TokenProbabilityResult":
    n = len(token_set)
    return TokenProbabilityResult(
        distribution={t: 1.0 / n for t in token_set},
        mass_capture=1.0,
        truncated_members=(),
    )


class FixedResponseLLM:
    """LLM mock that returns a fixed string for every generate call.

    Useful for tests that don't care about prompt content but want a
    specific response structure.
    """

    def __init__(self, response: str = "") -> None:
        self._response = response

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> str:
        return self._response

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Sequence[str]:
        return [self._response for _ in prompts]

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> "TokenProbabilityResult":
        n = len(token_set)
        return TokenProbabilityResult(
            distribution={t: 1.0 / n for t in token_set},
            mass_capture=1.0,
            truncated_members=(),
        )

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> Sequence["TokenProbabilityResult"]:
        n = len(token_set)
        return [
            TokenProbabilityResult(
                distribution={t: 1.0 / n for t in token_set},
                mass_capture=1.0,
                truncated_members=(),
            )
            for _ in prompts
        ]

    def get_metadata(self) -> Mapping[str, str]:
        return {"model": "FixedResponseLLM", "model_version": "1"}
