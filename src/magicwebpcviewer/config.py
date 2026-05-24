from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class AppConfig:
    data_dir: Path
    upload_dir: Path
    processed_dir: Path
    host: str = "127.0.0.1"
    port: int = 5006
    show_browser: bool = False
    use_xvfb: bool = False
    max_upload_mb: int = 512
    render_width: int = 1100
    render_height: int = 760
    max_render_points: int = 2_000_000
    websocket_origin: str | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = _path_from_env("PCV_DATA_DIR", Path("data/clouds"))
        upload_dir = _path_from_env("PCV_UPLOAD_DIR", data_dir / "uploads")
        processed_dir = _path_from_env("PCV_PROCESSED_DIR", data_dir / "processed")
        return cls(
            data_dir=data_dir,
            upload_dir=upload_dir,
            processed_dir=processed_dir,
            host=os.getenv("PCV_HOST", "127.0.0.1"),
            port=_int_from_env("PCV_PORT", 5006),
            show_browser=_bool_from_env("PCV_SHOW", False),
            use_xvfb=_bool_from_env("PCV_USE_XVFB", False),
            max_upload_mb=_int_from_env("PCV_MAX_UPLOAD_MB", 512),
            render_width=_int_from_env("PCV_RENDER_WIDTH", 1100),
            render_height=_int_from_env("PCV_RENDER_HEIGHT", 760),
            max_render_points=_int_from_env("PCV_MAX_RENDER_POINTS", 2_000_000),
            websocket_origin=os.getenv("PCV_WEBSOCKET_ORIGIN"),
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
