from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from .models import CloudData
from .processing import make_polydata


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RenderOptions:
    color_mode: str = "solid"
    scalar_field: str | None = None
    point_size: float = 3.0
    background: str = "#0b1020"
    solid_color: str = "#44a3ff"
    camera_view: str = "isometric"
    width: int = 1100
    height: int = 760
    max_render_points: int = 2_000_000


def maybe_start_xvfb(use_xvfb: bool) -> str | None:
    if not use_xvfb:
        return None
    try:
        import pyvista as pv

        pv.start_xvfb()
        return None
    except Exception as exc:
        return f"Could not start Xvfb: {exc}"


def render_cloud_to_png(cloud: CloudData, options: RenderOptions) -> bytes:
    try:
        import pyvista as pv

        render_cloud = _sample_for_render(cloud, options.max_render_points)
        poly = make_polydata(render_cloud)
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(int(options.width), int(options.height)),
            border=False,
        )
        plotter.set_background(options.background)

        add_kwargs = {
            "style": "points",
            "point_size": float(options.point_size),
            "render_points_as_spheres": True,
        }

        if options.color_mode == "rgb" and render_cloud.colors is not None:
            plotter.add_mesh(poly, scalars="RGB", rgb=True, **add_kwargs)
        elif options.color_mode == "scalar" and options.scalar_field in render_cloud.point_data:
            plotter.add_mesh(poly, scalars=options.scalar_field, cmap="viridis", **add_kwargs)
            plotter.add_scalar_bar(title=options.scalar_field)
        else:
            plotter.add_mesh(poly, color=options.solid_color, **add_kwargs)

        _apply_camera_view(plotter, options.camera_view)
        plotter.reset_camera()
        image = plotter.screenshot(return_img=True)
        plotter.close()
    except Exception as exc:
        raise RenderError(f"Server rendering failed: {exc}") from exc

    return _encode_png(image)


def _sample_for_render(cloud: CloudData, max_points: int) -> CloudData:
    if max_points <= 0 or cloud.point_count <= max_points:
        return cloud
    indices = np.linspace(0, cloud.point_count - 1, max_points, dtype=np.int64)
    point_data = {
        name: np.asarray(values)[indices]
        for name, values in cloud.point_data.items()
        if np.asarray(values).shape[:1] == (cloud.point_count,)
    }
    colors = None
    if cloud.colors is not None and cloud.colors.shape[:1] == (cloud.point_count,):
        colors = cloud.colors[indices]
    return CloudData(
        name=cloud.name,
        source_format=cloud.source_format,
        points=cloud.points[indices],
        path=cloud.path,
        file_size=cloud.file_size,
        point_data=point_data,
        colors=colors,
        color_name=cloud.color_name,
    )


def _apply_camera_view(plotter, camera_view: str) -> None:
    view = camera_view.lower()
    if view == "xy":
        plotter.view_xy()
    elif view == "xz":
        plotter.view_xz()
    elif view == "yz":
        plotter.view_yz()
    else:
        plotter.view_isometric()


def _encode_png(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
