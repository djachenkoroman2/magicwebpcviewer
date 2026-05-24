from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import panel as pn

from .config import AppConfig
from .ui import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Magic Web Point Cloud Viewer.")
    parser.add_argument("--host", help="Host/interface to bind. Default: PCV_HOST or 127.0.0.1.")
    parser.add_argument("--port", type=int, help="Port to bind. Default: PCV_PORT or 5006.")
    parser.add_argument("--data-dir", type=Path, help="Server directory with point clouds.")
    parser.add_argument("--upload-dir", type=Path, help="Directory where uploaded files are saved.")
    parser.add_argument("--processed-dir", type=Path, help="Directory where processed PLY files are saved.")
    parser.add_argument("--max-upload-mb", type=int, help="Maximum uploaded file size in MB.")
    parser.add_argument("--render-width", type=int, help="Server render width in pixels.")
    parser.add_argument("--render-height", type=int, help="Server render height in pixels.")
    parser.add_argument("--max-render-points", type=int, help="Maximum points used for preview rendering.")
    parser.add_argument("--websocket-origin", help="Comma-separated websocket origins for reverse proxy setups.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=None, help="Open browser on start.")
    parser.add_argument("--use-xvfb", action=argparse.BooleanOptionalAction, default=None, help="Start Xvfb before rendering.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.from_env()
    overrides = {}
    for attr in (
        "host",
        "port",
        "max_upload_mb",
        "render_width",
        "render_height",
        "max_render_points",
        "websocket_origin",
    ):
        value = getattr(args, attr)
        if value is not None:
            overrides[attr] = value
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir.expanduser().resolve()
    if args.upload_dir is not None:
        overrides["upload_dir"] = args.upload_dir.expanduser().resolve()
    if args.processed_dir is not None:
        overrides["processed_dir"] = args.processed_dir.expanduser().resolve()
    if args.show is not None:
        overrides["show_browser"] = args.show
    if args.use_xvfb is not None:
        overrides["use_xvfb"] = args.use_xvfb

    config = replace(config, **overrides)
    config.ensure_dirs()
    app = create_app(config)

    serve_kwargs = {
        "address": config.host,
        "port": config.port,
        "show": config.show_browser,
        "title": "Magic Web Point Cloud Viewer",
    }
    if config.websocket_origin:
        serve_kwargs["websocket_origin"] = [
            origin.strip() for origin in config.websocket_origin.split(",") if origin.strip()
        ]

    pn.serve(app, **serve_kwargs)


if __name__ == "__main__":
    main()
