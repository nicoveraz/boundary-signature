"""StateCanonicalizer protocol.

Hashes raw state representations into stable identifiers. The boundary
where domain-specific data becomes the framework's canonical
representation; getting it wrong silently produces garbage downstream.

Generic over ``RawT`` (contravariant) so domain implementations declare
the raw input type they accept (``EDStateCanonicalizer`` declares
``StateCanonicalizer[ClinicalEncounterRow]``) and ``mypy --strict``
verifies consumers pass the right type.

A canonicalizer's ``node_id`` namespace is scoped to a single
canonicalizer instance per AssemblyGraph. Mixing node_ids from different
canonicalizers in one graph is undefined behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

RawT = TypeVar("RawT", contravariant=True)


class StateCanonicalizer(Protocol[RawT]):
    def canonicalize(self, raw_state: RawT) -> tuple[str, str]:
        """Return ``(stable_hash, human_readable_form)``.

        Equivalent raw states MUST produce the same ``stable_hash``.
        On unrecoverable failure, raise ``CanonicalizationError``.
        """
        ...

    def get_metadata(self) -> Mapping[str, str]:
        """Canonicalizer name, version, config hash for reproducibility."""
        ...
