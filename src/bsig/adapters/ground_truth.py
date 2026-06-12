"""GroundTruthExtractor protocol.

Produces Outcome objects from raw trajectory data. Multi-signal weak
supervision logic lives in implementations, not in core.

Generic over ``RawTrajT`` (contravariant), parallel to
``StateCanonicalizer``: the extractor's input is domain-specific raw
data (DB rows, dataset records) and ``Any`` would kill type checking at
exactly the boundary most likely to be misused. Concrete implementations
declare the raw type they accept (e.g., ``ClinicalGroundTruthExtractor``
declares ``GroundTruthExtractor[ClinicalEncounterRow]``) and
``mypy --strict`` verifies consumers pass the right type.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

from bsig.core.trajectory import Outcome

RawTrajT = TypeVar("RawTrajT", contravariant=True)


class GroundTruthExtractor(Protocol[RawTrajT]):
    def extract(self, raw_trajectory: RawTrajT) -> Outcome | None:
        """Return Outcome if ground truth is available; None otherwise."""
        ...

    def get_metadata(self) -> Mapping[str, str]:
        """Extractor name, version, signal-set version for reproducibility."""
        ...
