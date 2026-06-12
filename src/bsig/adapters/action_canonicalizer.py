"""ActionCanonicalizer protocol.

Hashes raw action representations into stable identifiers. Parallel
to ``StateCanonicalizer``: the boundary where domain-specific action
data (reasoning step text + position for MedQA; test order + ATC
class for clinical) becomes the framework's canonical ``action_id``.

Generic over ``RawActionT`` (contravariant) so domain implementations
declare the raw input type they accept (e.g.,
``MCQActionCanonicalizer`` declares
``ActionCanonicalizer[ReasoningStepRawAction]``) and ``mypy --strict``
verifies consumers pass the right type.

**Architectural note (per stage 3.3b design pass):**
Canonicalization is about identity, not behavior. Two trajectories
with content-similar actions get the same ``action_id``; their
distribution-shift behavior is captured at the recovered-graph's
edge level (via VoI on edges in ``recovery.py``), NOT by encoding
behavior into the action_id. Making the canonicalizer "shift-aware"
(taking source/target distributions as input) would canonicalize the
framework's signal away — see
``docs/exploration/2026-05-03-internal-disagreement-as-headline-claim.md``
for the full reasoning.

A canonicalizer's ``action_id`` namespace is scoped to a single
canonicalizer instance per AssemblyGraph, parallel to
``StateCanonicalizer``. Mixing action_ids from different
canonicalizers in one graph is undefined behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

RawActionT = TypeVar("RawActionT", contravariant=True)


class ActionCanonicalizer(Protocol[RawActionT]):
    def canonicalize(self, raw_action: RawActionT) -> tuple[str, str]:
        """Return ``(stable_action_id, human_readable_form)``.

        Equivalent raw actions MUST produce the same ``stable_action_id``.
        On unrecoverable failure, raise ``CanonicalizationError``.
        """
        ...

    def get_metadata(self) -> Mapping[str, str]:
        """Canonicalizer name, version, config hash for reproducibility."""
        ...
