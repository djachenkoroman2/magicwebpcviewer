from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CloudData:
    """In-memory point cloud representation used by the app."""

    name: str
    source_format: str
    points: np.ndarray
    path: Path | None = None
    file_size: int = 0
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    colors: np.ndarray | None = None
    color_name: str | None = None

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    def with_name(self, name: str, path: Path | None = None) -> "CloudData":
        return CloudData(
            name=name,
            source_format=self.source_format,
            points=self.points,
            path=path if path is not None else self.path,
            file_size=self.file_size,
            point_data=self.point_data,
            colors=self.colors,
            color_name=self.color_name,
        )
