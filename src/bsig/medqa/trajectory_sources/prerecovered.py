"""Pre-recovered trajectory source.

Reads cached Trajectory objects from a stage-3.2 cached-trajectories
artifact directory (see ``serialization.py`` for the format). Satisfies
``TrajectorySource`` from ``bsig.adapters``.

Use case: re-running signature analysis with different
``SignatureWeights`` against the same Condition C output without
re-issuing the LLM calls. The expensive work (LLM trajectory
generation) is done once; signature scoring becomes cheap to iterate.

The writer (Condition C's output persistence) lands in stage 3.3.
This stage-3.2 reader exists with format-locked schema; an integration
test verifies round-trip via synthetic Trajectories.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from bsig.core.trajectory import Trajectory
from bsig.medqa.trajectory_sources.serialization import (
    iter_cached_trajectories,
    load_cached_trajectories,
)


class MedQAPrerecoveredTrajectorySource:
    """Satisfies ``TrajectorySource`` against a cached-trajectories
    artifact directory.

    Single-shot iterator (re-instantiate to re-iterate, parallel to
    other ``TrajectorySource`` implementations).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def iter_trajectories(self) -> Iterator[Trajectory]:
        return iter_cached_trajectories(self._path)

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            "source_name": "MedQAPrerecoveredTrajectorySource",
            "source_version": "1",
            "artifact_path": str(self._path),
        }

    def load_all(self) -> list[Trajectory]:
        """Materialize all trajectories at once. Convenience for
        callers that need the full list (e.g., recovery, which takes
        ``Sequence[Trajectory]``)."""
        return load_cached_trajectories(self._path)
