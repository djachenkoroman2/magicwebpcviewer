from __future__ import annotations

import base64
import html
from dataclasses import replace
from pathlib import Path

import panel as pn

from .cloud_io import (
    CloudLoadError,
    list_cloud_files,
    load_point_cloud,
    resolve_server_file,
    safe_filename,
    save_point_cloud_as_ply,
    unique_path,
)
from .config import AppConfig
from .models import CloudData
from .processing import available_scalar_fields, compute_properties, voxel_downsample
from .visualization import RenderOptions, maybe_start_xvfb, render_cloud_to_png


class PointCloudViewerApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_dirs()
        self.current_cloud: CloudData | None = None
        self.last_downsample_text = ""

        pn.extension(sizing_mode="stretch_width")
        xvfb_warning = maybe_start_xvfb(config.use_xvfb)

        self.file_select = pn.widgets.Select(name="Файл на сервере", options=[])
        self.refresh_button = pn.widgets.Button(name="Обновить", button_type="light")
        self.open_button = pn.widgets.Button(name="Открыть", button_type="primary")
        self.clear_button = pn.widgets.Button(name="Очистить", button_type="warning")

        self.upload_input = pn.widgets.FileInput(
            name="Загрузить файл",
            accept=".ply,.pcd,.las,.laz,.xyz,.txt,.csv,.vtk,.vtp",
        )
        self.upload_downsample = pn.widgets.Checkbox(name="Даунсемплинг при загрузке", value=False)
        self.upload_button = pn.widgets.Button(name="Загрузить на сервер", button_type="primary")

        self.voxel_size = pn.widgets.FloatInput(name="Размер voxel", value=0.1, step=0.05, start=0.000001)
        self.downsample_button = pn.widgets.Button(name="Даунсемплинг", button_type="primary")
        self.download_button = pn.widgets.FileDownload(
            label="Скачать PLY",
            button_type="success",
            disabled=True,
            filename="cloud.ply",
            file=None,
        )

        self.color_mode = pn.widgets.RadioButtonGroup(
            name="Режим цвета",
            options={"RGB": "rgb", "Scalar": "scalar", "Однотонно": "solid"},
            value="solid",
            button_type="primary",
        )
        self.scalar_select = pn.widgets.Select(name="Scalar field", options=[])
        self.point_size = pn.widgets.FloatSlider(name="Размер точки", value=3.0, start=1.0, end=12.0, step=0.5)
        self.background = pn.widgets.ColorPicker(name="Фон", value="#0b1020")
        self.solid_color = pn.widgets.ColorPicker(name="Цвет", value="#44a3ff")
        self.camera_view = pn.widgets.Select(
            name="Камера",
            options={"Изометрия": "isometric", "XY": "xy", "XZ": "xz", "YZ": "yz"},
            value="isometric",
        )
        self.reset_camera_button = pn.widgets.Button(name="Сброс камеры", button_type="light")

        self.viewer = pn.pane.HTML(self._empty_view(), height=680, sizing_mode="stretch_width")
        self.properties = pn.pane.HTML(self._properties_placeholder(), sizing_mode="stretch_width")
        self.status = pn.pane.HTML("", sizing_mode="stretch_width")

        self._wire_events()
        self.refresh_file_list()
        if xvfb_warning:
            self._set_status(xvfb_warning, level="warning")

    def view(self):
        sidebar = pn.Column(
            "### Файлы",
            self.file_select,
            pn.Row(self.open_button, self.refresh_button),
            self.upload_input,
            self.upload_downsample,
            self.upload_button,
            "### Операции",
            self.voxel_size,
            self.downsample_button,
            self.download_button,
            self.clear_button,
            "### Визуализация",
            self.color_mode,
            self.scalar_select,
            pn.Row(self.point_size),
            pn.Row(self.background, self.solid_color),
            self.camera_view,
            self.reset_camera_button,
            sizing_mode="stretch_width",
        )
        main = pn.Column(
            self.status,
            self.viewer,
            pn.layout.Divider(),
            "### Свойства облака",
            self.properties,
            sizing_mode="stretch_both",
        )
        template = pn.template.FastListTemplate(
            title="Magic Web Point Cloud Viewer",
            sidebar=[sidebar],
            main=[main],
            accent_base_color="#2563eb",
            header_background="#111827",
            main_max_width="none",
        )
        return template

    def refresh_file_list(self, event=None) -> None:
        files = list_cloud_files(self.config.data_dir)
        self.file_select.options = files
        if files and self.file_select.value not in files:
            self.file_select.value = files[0]
        if not files:
            self.file_select.value = None

    def open_selected_file(self, event=None) -> None:
        if not self.file_select.value:
            self._set_status("В директории данных нет выбранного файла.", level="warning")
            return
        try:
            path = resolve_server_file(self.config.data_dir, self.file_select.value)
            cloud = load_point_cloud(path)
            self._set_current_cloud(cloud, f"Открыт файл {cloud.name}.")
        except CloudLoadError as exc:
            self._set_status(str(exc), level="error")

    def upload_file(self, event=None) -> None:
        if not self.upload_input.value:
            self._set_status("Выберите файл для загрузки.", level="warning")
            return
        payload = self.upload_input.value
        if len(payload) > self.config.max_upload_bytes:
            self._set_status(
                f"Файл больше лимита {self.config.max_upload_mb} MB.",
                level="error",
            )
            return

        filename = safe_filename(self.upload_input.filename or "cloud")
        suffix = Path(filename).suffix.lower()
        if not suffix:
            self._set_status("У загружаемого файла нет расширения.", level="error")
            return

        try:
            self.config.upload_dir.mkdir(parents=True, exist_ok=True)
            destination = unique_path(self.config.upload_dir / filename)
            destination.write_bytes(payload)
            cloud = load_point_cloud(destination)
            message = f"Файл {destination.name} загружен полностью."
            if self.upload_downsample.value:
                result = voxel_downsample(cloud, float(self.voxel_size.value))
                cloud = result.cloud.with_name(f"{destination.stem}_downsampled")
                exported = self._export_cloud(cloud, f"{destination.stem}_downsampled.ply")
                cloud = replace(cloud, path=exported, source_format="ply", file_size=exported.stat().st_size)
                message = self._downsample_message(result, prefix=f"Файл {destination.name} загружен с даунсемплингом.")
            self.refresh_file_list()
            self._set_current_cloud(cloud, message)
        except Exception as exc:
            self._set_status(f"Ошибка загрузки: {exc}", level="error")

    def clear_scene(self, event=None) -> None:
        self.current_cloud = None
        self.viewer.object = self._empty_view()
        self.properties.object = self._properties_placeholder()
        self.download_button.disabled = True
        self.download_button.file = None
        self.last_downsample_text = ""
        self._set_status("Сцена очищена.")

    def downsample_current(self, event=None) -> None:
        if self.current_cloud is None:
            self._set_status("Сначала откройте облако точек.", level="warning")
            return
        try:
            result = voxel_downsample(self.current_cloud, float(self.voxel_size.value))
            cloud = result.cloud
            exported = self._export_cloud(cloud, f"{safe_filename(cloud.name)}.ply")
            cloud = replace(cloud, path=exported, source_format="ply", file_size=exported.stat().st_size)
            message = self._downsample_message(result)
            self._set_current_cloud(cloud, message)
            self.refresh_file_list()
        except Exception as exc:
            self._set_status(f"Ошибка даунсемплинга: {exc}", level="error")

    def reset_camera(self, event=None) -> None:
        self.camera_view.value = "isometric"
        self.render_current()

    def render_current(self, event=None) -> None:
        if self.current_cloud is None:
            return
        try:
            options = RenderOptions(
                color_mode=self.color_mode.value,
                scalar_field=self.scalar_select.value,
                point_size=float(self.point_size.value),
                background=self.background.value,
                solid_color=self.solid_color.value,
                camera_view=self.camera_view.value,
                width=self.config.render_width,
                height=self.config.render_height,
                max_render_points=self.config.max_render_points,
            )
            image_bytes = render_cloud_to_png(self.current_cloud, options)
            self.viewer.object = self._image_html(image_bytes)
            if self.current_cloud.point_count > self.config.max_render_points:
                self._set_status(
                    f"Для рендера показана равномерная выборка {self.config.max_render_points:,} точек из {self.current_cloud.point_count:,}.",
                    level="warning",
                )
        except Exception as exc:
            self._set_status(str(exc), level="error")

    def _wire_events(self) -> None:
        self.refresh_button.on_click(self.refresh_file_list)
        self.open_button.on_click(self.open_selected_file)
        self.upload_button.on_click(self.upload_file)
        self.clear_button.on_click(self.clear_scene)
        self.downsample_button.on_click(self.downsample_current)
        self.reset_camera_button.on_click(self.reset_camera)
        for widget in (
            self.color_mode,
            self.scalar_select,
            self.point_size,
            self.background,
            self.solid_color,
            self.camera_view,
        ):
            widget.param.watch(self.render_current, "value")

    def _set_current_cloud(self, cloud: CloudData, message: str) -> None:
        self.current_cloud = cloud
        scalar_fields = available_scalar_fields(cloud)
        self.scalar_select.options = scalar_fields
        self.scalar_select.value = scalar_fields[0] if scalar_fields else None
        if cloud.colors is not None:
            self.color_mode.value = "rgb"
        elif scalar_fields:
            self.color_mode.value = "scalar"
        else:
            self.color_mode.value = "solid"
        self.properties.object = self._properties_html(cloud)
        self._set_status(message)
        self._prepare_download(cloud)
        self.render_current()

    def _prepare_download(self, cloud: CloudData) -> None:
        try:
            if cloud.path is not None and cloud.path.exists() and cloud.path.suffix.lower() == ".ply":
                exported = cloud.path
            else:
                exported = self._export_cloud(cloud, f"{safe_filename(cloud.name)}.ply")
            self.download_button.file = str(exported)
            self.download_button.filename = exported.name
            self.download_button.disabled = False
        except Exception as exc:
            self.download_button.disabled = True
            self.download_button.file = None
            self._set_status(f"Не удалось подготовить скачивание: {exc}", level="error")

    def _export_cloud(self, cloud: CloudData, filename: str) -> Path:
        safe_name = safe_filename(filename)
        destination = unique_path(self.config.processed_dir / safe_name)
        return save_point_cloud_as_ply(cloud, destination)

    def _downsample_message(self, result, prefix: str = "Даунсемплинг выполнен.") -> str:
        reduction = result.reduction_ratio * 100
        self.last_downsample_text = (
            f"{prefix} Было {result.original_count:,}, стало {result.downsampled_count:,}, "
            f"сохранено {reduction:.1f}% точек."
        )
        return self.last_downsample_text

    def _set_status(self, text: str, level: str = "info") -> None:
        colors = {
            "info": "#dbeafe",
            "warning": "#fef3c7",
            "error": "#fee2e2",
        }
        text_colors = {
            "info": "#1e3a8a",
            "warning": "#92400e",
            "error": "#991b1b",
        }
        background = colors.get(level, colors["info"])
        foreground = text_colors.get(level, text_colors["info"])
        self.status.object = (
            f"<div style='background:{background};color:{foreground};"
            "padding:10px 12px;border-radius:6px;font-weight:600;'>"
            f"{html.escape(text)}</div>"
        )

    def _properties_html(self, cloud: CloudData) -> str:
        props = compute_properties(cloud)
        scalar_fields = props["scalar_fields"]
        rows = [
            ("Имя", props["name"]),
            ("Формат", props["format"]),
            ("Размер файла", _format_bytes(int(props["file_size"]))),
            ("Количество точек", f"{int(props['point_count']):,}"),
            ("RGB", "есть" if props["has_rgb"] else "нет"),
            ("Scalar fields", ", ".join(scalar_fields) if scalar_fields else "нет"),
            ("Bounding box min", _format_vector(props["bounds_min"])),
            ("Bounding box max", _format_vector(props["bounds_max"])),
            ("Размер bbox", _format_vector(props["bounds_size"])),
            ("Плотность", _format_density(props["density"])),
        ]
        if self.last_downsample_text:
            rows.append(("Последняя операция", self.last_downsample_text))
        body = "".join(
            "<tr>"
            f"<th>{html.escape(str(name))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
            for name, value in rows
        )
        return (
            "<table style='width:100%;border-collapse:collapse;'>"
            "<style>th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;}"
            "th{width:220px;color:#374151;background:#f9fafb;}</style>"
            f"{body}</table>"
        )

    def _properties_placeholder(self) -> str:
        return "<div style='color:#6b7280;padding:8px 0;'>Облако точек не загружено.</div>"

    def _empty_view(self) -> str:
        return (
            "<div style='height:100%;min-height:620px;display:flex;align-items:center;"
            "justify-content:center;background:#0b1020;color:#cbd5e1;border-radius:6px;'>"
            "Откройте или загрузите облако точек</div>"
        )

    def _image_html(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return (
            "<div style='width:100%;height:100%;min-height:620px;background:#111827;"
            "display:flex;align-items:center;justify-content:center;border-radius:6px;overflow:hidden;'>"
            f"<img src='data:image/png;base64,{encoded}' "
            "style='max-width:100%;max-height:100%;object-fit:contain;'/>"
            "</div>"
        )


def _format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size} B"


def _format_vector(values) -> str:
    return ", ".join(f"{float(value):.4g}" for value in values)


def _format_density(value) -> str:
    if value is None:
        return "недоступно"
    return f"{value:.4g} точек/ед. объема"


def create_app(config: AppConfig | None = None):
    return PointCloudViewerApp(config or AppConfig.from_env()).view()
