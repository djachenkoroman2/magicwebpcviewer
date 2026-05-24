from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np

from .models import CloudData
from .processing import make_polydata, normalize_colors, validate_points

SUPPORTED_EXTENSIONS = {
    ".ply",
    ".pcd",
    ".las",
    ".laz",
    ".xyz",
    ".txt",
    ".csv",
    ".vtk",
    ".vtp",
}


class CloudLoadError(RuntimeError):
    pass


def list_cloud_files(root: Path) -> list[str]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []

    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def resolve_server_file(root: Path, relative_path: str) -> Path:
    root = root.expanduser().resolve()
    path = (root / relative_path).expanduser().resolve()
    if root != path and root not in path.parents:
        raise CloudLoadError("Selected path is outside the configured data directory.")
    return path


def load_point_cloud(path: Path) -> CloudData:
    path = path.expanduser().resolve()
    if not path.exists():
        raise CloudLoadError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise CloudLoadError(f"Unsupported file format: {path.suffix}")

    suffix = path.suffix.lower()
    try:
        if suffix in {".xyz", ".txt", ".csv"}:
            return _read_table_cloud(path)
        if suffix in {".las", ".laz"}:
            return _read_las_cloud(path)
        if suffix == ".pcd":
            try:
                return _read_pyvista_cloud(path)
            except Exception:
                return _read_ascii_pcd(path)
        return _read_pyvista_cloud(path)
    except CloudLoadError:
        raise
    except Exception as exc:
        raise CloudLoadError(f"Could not read {path.name}: {exc}") from exc


def save_point_cloud_as_ply(cloud: CloudData, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".ply":
        path = path.with_suffix(".ply")
    poly = make_polydata(cloud)
    poly.save(path)
    return path


def unique_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {path}")


def safe_filename(filename: str) -> str:
    name = Path(filename or "cloud").name.strip()
    if not name:
        name = "cloud"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    return cleaned or "cloud"


def _read_table_cloud(path: Path) -> CloudData:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        raw = np.loadtxt(path, comments="#", delimiter=delimiter)
    except ValueError as exc:
        raise CloudLoadError(f"Could not parse numeric table: {exc}") from exc

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        raise CloudLoadError("Table point cloud must contain at least XYZ columns.")

    points = validate_points(raw[:, :3])
    tail = raw[:, 3:]
    colors = None
    point_data: dict[str, np.ndarray] = {}

    if tail.shape[1] >= 3 and _looks_like_rgb(tail[:, :3]):
        colors = normalize_colors(tail[:, :3])
        for index in range(3, tail.shape[1]):
            point_data[f"field_{index - 2}"] = tail[:, index]
    else:
        for index in range(tail.shape[1]):
            point_data[f"field_{index + 1}"] = tail[:, index]

    return CloudData(
        name=path.name,
        source_format=path.suffix.lower().lstrip("."),
        points=points,
        path=path,
        file_size=path.stat().st_size,
        point_data=point_data,
        colors=colors,
        color_name="RGB" if colors is not None else None,
    )


def _looks_like_rgb(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    min_value = float(finite.min())
    max_value = float(finite.max())
    return min_value >= 0 and max_value <= 255


def _read_las_cloud(path: Path) -> CloudData:
    try:
        import laspy
    except ImportError as exc:
        raise CloudLoadError("LAS/LAZ support requires the laspy package.") from exc

    las = laspy.read(path)
    points = validate_points(np.column_stack((las.x, las.y, las.z)))
    point_count = points.shape[0]
    point_data: dict[str, np.ndarray] = {}

    colors = None
    if all(hasattr(las, name) for name in ("red", "green", "blue")):
        raw_colors = np.column_stack((las.red, las.green, las.blue))
        if raw_colors.shape[0] == point_count:
            colors = normalize_colors(raw_colors)

    skip_names = {"x", "y", "z", "red", "green", "blue"}
    for name in las.point_format.dimension_names:
        if name.lower() in skip_names:
            continue
        try:
            values = np.asarray(las[name])
        except Exception:
            continue
        if values.shape[:1] == (point_count,) and np.issubdtype(values.dtype, np.number):
            point_data[name] = values

    return CloudData(
        name=path.name,
        source_format=path.suffix.lower().lstrip("."),
        points=points,
        path=path,
        file_size=path.stat().st_size,
        point_data=point_data,
        colors=colors,
        color_name="RGB" if colors is not None else None,
    )


def _read_pyvista_cloud(path: Path) -> CloudData:
    import pyvista as pv

    dataset = pv.read(path)
    if isinstance(dataset, pv.MultiBlock):
        dataset = dataset.combine()
    if not hasattr(dataset, "points") or dataset.points is None:
        raise CloudLoadError("Dataset does not contain point coordinates.")

    points = validate_points(np.asarray(dataset.points))
    point_count = points.shape[0]
    raw_point_data = getattr(dataset, "point_data", {})
    colors, color_name = _extract_colors(raw_point_data, point_count)
    point_data = _extract_scalar_fields(raw_point_data, point_count, color_name)

    return CloudData(
        name=path.name,
        source_format=path.suffix.lower().lstrip("."),
        points=points,
        path=path,
        file_size=path.stat().st_size,
        point_data=point_data,
        colors=colors,
        color_name=color_name,
    )


def _extract_colors(point_data, point_count: int) -> tuple[np.ndarray | None, str | None]:
    preferred = ("RGB", "RGBA", "rgb", "rgba", "Colors", "colors")
    names = list(point_data.keys())
    for name in preferred + tuple(names):
        if name not in point_data:
            continue
        array = np.asarray(point_data[name])
        if array.shape[:1] != (point_count,):
            continue
        if array.ndim == 2 and array.shape[1] in {3, 4} and np.issubdtype(array.dtype, np.number):
            try:
                return normalize_colors(array), name
            except ValueError:
                continue
        if name.lower() in {"rgb", "rgba"} and array.ndim == 1 and np.issubdtype(array.dtype, np.number):
            try:
                return normalize_colors(array), name
            except ValueError:
                continue
    return None, None


def _extract_scalar_fields(point_data, point_count: int, color_name: str | None) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in point_data.keys():
        if color_name is not None and name == color_name:
            continue
        array = np.asarray(point_data[name])
        if array.shape[:1] != (point_count,):
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        if array.ndim == 1:
            fields[name] = array
        elif array.ndim == 2 and array.shape[1] == 1:
            fields[name] = array[:, 0]
        elif array.ndim == 2 and array.shape[1] <= 4:
            for index in range(array.shape[1]):
                fields[f"{name}_{index + 1}"] = array[:, index]
    return fields


def _read_ascii_pcd(path: Path) -> CloudData:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    header: dict[str, str] = {}
    data_start = None
    data_type = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        key = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""
        if key == "DATA":
            data_start = index + 1
            data_type = value.lower()
            break
        header[key] = value

    if data_start is None or data_type is None:
        raise CloudLoadError("PCD file does not contain a DATA section.")
    if data_type != "ascii":
        raise CloudLoadError("Only ASCII PCD fallback parsing is supported when PyVista cannot read the file.")

    fields = header.get("FIELDS", "").split()
    if not fields:
        raise CloudLoadError("PCD file does not declare fields.")
    lower_fields = [field.lower() for field in fields]
    for required in ("x", "y", "z"):
        if required not in lower_fields:
            raise CloudLoadError("PCD file must contain x, y and z fields.")

    data_text = "\n".join(lines[data_start:]).strip()
    if not data_text:
        raise CloudLoadError("PCD file has no point records.")
    raw = np.loadtxt(StringIO(data_text))
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    columns = {field: raw[:, index] for index, field in enumerate(lower_fields)}
    points = validate_points(np.column_stack((columns["x"], columns["y"], columns["z"])))
    colors = _pcd_colors(columns)

    point_data: dict[str, np.ndarray] = {}
    skip = {"x", "y", "z", "rgb", "rgba", "r", "g", "b"}
    for field, values in columns.items():
        if field in skip:
            continue
        point_data[field] = values

    return CloudData(
        name=path.name,
        source_format="pcd",
        points=points,
        path=path,
        file_size=path.stat().st_size,
        point_data=point_data,
        colors=colors,
        color_name="RGB" if colors is not None else None,
    )


def _pcd_colors(columns: dict[str, np.ndarray]) -> np.ndarray | None:
    if all(channel in columns for channel in ("r", "g", "b")):
        return normalize_colors(np.column_stack((columns["r"], columns["g"], columns["b"])))
    if "rgb" in columns:
        return normalize_colors(columns["rgb"])
    if "rgba" in columns:
        return normalize_colors(columns["rgba"])
    return None
