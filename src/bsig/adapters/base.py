"""Shared types and exceptions for adapter contracts.

The exception hierarchy is part of the adapter contract surface. Concrete
implementations raise these types; downstream code catches them.
"""
from __future__ import annotations

from collections.abc import Mapping

AdapterMetadata = Mapping[str, str]


class AdapterError(Exception):
    """Base class for adapter-related errors."""


class LLMAdapterError(AdapterError):
    """Raised when an LLM adapter fails irrecoverably.

    For batch methods (``generate_batch`` and
    ``get_token_probabilities_batch``), this is raised when any item
    exhausts its per-item retries.

    Two optional fields enable downstream **surgical repair** — re-
    issuing only the failed item rather than re-running the entire
    batch:

    - ``failed_index``: the position of the item that exhausted
      retries. ``None`` for single-item operations or for batch
      failures whose origin can't be localized.
    - ``partial_results``: a sequence of length ``len(prompts)``
      carrying successful items at non-failed positions and ``None`` at
      failed and not-yet-attempted positions. Element type depends on
      the failing batch operation:

      * For ``generate_batch`` failures:
        ``Sequence[str | None]`` (str at successful positions).
      * For ``get_token_probabilities_batch`` failures:
        ``Sequence[TokenProbabilityResult | None]``.

      ``None`` at the partial_results level (rather than per-position)
      when the implementation can't preserve partial state. Typed as
      ``object`` on the exception to accommodate both shapes; callers
      cast at use sites based on which batch method failed.

    Adopters that populate these fields enable Condition C to avoid
    re-issuing successful items on repair (saves ~4 LLM calls per
    failure at the typical 5-item-batch case). Adopters that don't
    populate them get atomic-repair fallback in Condition C — the
    full batch is re-issued.
    """

    def __init__(
        self,
        *args: object,
        failed_index: int | None = None,
        partial_results: object = None,
    ) -> None:
        super().__init__(*args)
        self.failed_index = failed_index
        self.partial_results = partial_results


class CanonicalizationError(AdapterError):
    """Raised when state canonicalization fails."""
