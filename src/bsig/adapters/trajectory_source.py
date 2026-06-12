"""TrajectorySource protocol.

Iterator over Trajectory objects. Single-shot by design: if a consumer
needs to re-iterate, the source is re-instantiated. This avoids the
"is this iterator already exhausted?" failure mode.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol

from bsig.core.trajectory import Trajectory


class TrajectorySource(Protocol):
    def iter_trajectories(self) -> Iterator[Trajectory]:
        """Yield trajectories. Single-shot; re-instantiate to re-iterate."""
        ...

    def get_metadata(self) -> Mapping[str, Any]:
        """Source description, dataset version, etc."""
        ...
