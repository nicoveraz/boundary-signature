"""bsig.reference — production-ready reference adapter implementations.

Each implementation sits behind an opt-in extra in pyproject.toml so
core bsig has minimal required dependencies:

- ``OllamaLLMAdapter`` (``ollama`` extra: httpx): text-generation
  LLMAdapter for Ollama-served models. Implements ``generate`` and
  ``generate_batch`` only; Ollama's API does not expose next-token
  logprobs, so this adapter does not support
  ``get_token_probabilities``. Used for the smoke-test path.
- ``MLXLLMAdapter`` (``mlx`` extra: mlx-lm — Phase A scaffolding):
  Apple Silicon native LLMAdapter via direct mlx-lm. Implements the
  full Protocol surface plus
  ``generate_completions_with_shared_prefix`` for Phase C semantic
  entropy. Numerical-validation status: not yet certified;
  cross-adapter agreement test (§7.1 of stage_6_mlx_adapter
  pre-design) is the gate before results computed under MLX are
  reportable as framework findings.
- ``LlamaCppLLMAdapter`` (``llama_cpp`` extra: httpx): primary
  LLMAdapter under ADR-0008's unified-measurement protocol. Connects
  to a llama.cpp server's OpenAI-compatible API; exposes
  ``get_token_probabilities`` returning a TokenProbabilityResult with
  the renormalised conditional distribution and mass-capture fraction.
- ``SentenceTransformerEmbedder`` (``sentence-transformers`` extra):
  EmbeddingSource backed by sentence-transformers, defaulting to
  ``intfloat/multilingual-e5-large`` per CLAUDE.md §8.
- (later) ``OpenAILLMAdapter`` (``openai`` extra): OpenAI-API-compatible
  client for vLLM-served models on H100.

Reference implementations are available to anyone using the framework;
they are not the only valid implementations. Downstream users can
write their own adapters without touching this code.
"""
from __future__ import annotations

from bsig.reference.embedding_st import SentenceTransformerEmbedder
from bsig.reference.llm_llama_cpp import LlamaCppLLMAdapter
from bsig.reference.llm_local import OllamaLLMAdapter
from bsig.reference.llm_mlx import MLXLLMAdapter

__all__ = [
    "LlamaCppLLMAdapter",
    "MLXLLMAdapter",
    "OllamaLLMAdapter",
    "SentenceTransformerEmbedder",
]
