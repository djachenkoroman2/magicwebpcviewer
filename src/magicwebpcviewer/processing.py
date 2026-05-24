from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CloudData


@dataclass(frozen=True, slots=True)
class DownsampleResult:
    cloud: CloudData
    original_count: int
    downsampled_count: int

    @property
    def reduction_ratio(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.downsampled_count / self.original_count


def validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point data must be a two-dimensional array with at least 3 columns.")
    points = points[:, :3]
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty.")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite XYZ coordinates.")
    return points


def normalize_colors(values: np.ndarray) -> np.ndarray:
    colors = np.asarray(values)
    if colors.ndim == 1:
        colors = _unpack_packed_rgb(colors)
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError("Color array must have RGB or RGBA columns.")

    colors = colors[:, :3]
    if np.issubdtype(colors.dtype, np.floating):
        max_value = float(np.nanmax(colors)) if colors.size else 0.0
        if max_value <= 1.0:
            colors = colors * 255.0
        elif max_value > 255.0:
            colors = colors / 65535.0 * 255.0
    elif colors.size and int(np.nanmax(colors)) > 255:
        colors = colors.astype(np.float64) / 65535.0 * 255.0

    return np.clip(colors, 0, 255).astype(np.uint8)


def _unpack_packed_rgb(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.floating):
        packed = raw.astype(np.float32).view(np.uint32)
    else:
        packed = raw.astype(np.uint32)
    red = (packed >> 16) & 255
    green = (packed >> 8) & 255
    blue = packed & 255
    return np.column_stack((red, green, blue)).astype(np.uint8)


def available_scalar_fields(cloud: CloudData) -> list[str]:
    fields: list[str] = []
    for name, values in sorted(cloud.point_data.items()):
        array = np.asarray(values)
        if array.shape[:1] != (cloud.point_count,):
            continue
        if array.ndim != 1:
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        fields.append(name)
    return fields


def compute_properties(cloud: CloudData) -> dict[str, object]:
    points = cloud.points
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    volume = float(np.prod(span)) if np.all(span > 0) else 0.0
    density = float(cloud.point_count / volume) if volume > 0 else None
    return {
        "name": cloud.name,
        "format": cloud.source_format,
        "file_size": cloud.file_size,
        "point_count": cloud.point_count,
        "has_rgb": cloud.colors is not None,
        "scalar_fields": available_scalar_fields(cloud),
        "bounds_min": mins,
        "bounds_max": maxs,
        "bounds_size": span,
        "density": density,
    }


def voxel_downsample(cloud: CloudData, voxel_size: float) -> DownsampleResult:
    if voxel_size <= 0:
        raise ValueError("Voxel size must be greater than zero.")

    points = validate_points(cloud.points)
    origin = points.min(axis=0)
    voxel_index = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_index, axis=0, return_index=True)
    unique_indices.sort()

    reduced_points = points[unique_indices]
    reduced_point_data: dict[str, np.ndarray] = {}
    for name, values in cloud.point_data.items():
        array = np.asarray(values)
        if array.shape[:1] == (cloud.point_count,):
            reduced_point_data[name] = array[unique_indices]

    reduced_colors = None
    if cloud.colors is not None and cloud.colors.shape[:1] == (cloud.point_count,):
        reduced_colors = cloud.colors[unique_indices]

    reduced = CloudData(
        name=f"{cloud.name}_voxel_{voxel_size:g}",
        source_format=cloud.source_format,
        points=reduced_points,
        path=cloud.path,
        file_size=cloud.file_size,
        point_data=reduced_point_data,
        colors=reduced_colors,
        color_name=cloud.color_name,
    )
    return DownsampleResult(
        cloud=reduced,
        original_count=cloud.point_count,
        downsampled_count=reduced.point_count,
    )


def make_polydata(cloud: CloudData):
    import pyvista as pv

    poly = pv.PolyData(cloud.points)
    for name, values in cloud.point_data.items():
        array = np.asarray(values)
        if array.shape[:1] == (cloud.point_count,):
            poly.point_data[name] = array
    if cloud.colors is not None and cloud.colors.shape[:1] == (cloud.point_count,):
        poly.point_data["RGB"] = normalize_colors(cloud.colors)
    return poly
