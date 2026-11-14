"""Command-line entry point for the Phase 1 local simulation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from visionbridge import __version__
from visionbridge.core.configuration import ConfigurationError, load_settings


def build_parser() -> argparse.ArgumentParser:
    """Describe only available Phase 1 capabilities."""
    parser = argparse.ArgumentParser(
        description="VisionBridge Phase 1: local simulation, without AI inference or audio."
    )
    parser.add_argument("--version", action="version", version=f"VisionBridge {__version__}")
    parser.add_argument("--config", type=Path, help="YAML path; omitted means built-in defaults")
    parser.add_argument("--source", choices=("sample", "image", "webcam"))
    parser.add_argument("--image", type=Path, help="Image path (selects image unless --source is given)")
    parser.add_argument("--camera-index", type=int, help="Webcam index (selects webcam unless --source is given)")
    parser.add_argument("--max-frames", type=int, help="Positive frame limit (default: 1)")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None,
                        help="Open a local preview window; requires a desktop display")
    parser.add_argument("--output", type=Path, help="Save the last frame to a local image path")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return 0 on success, 1 on an expected failure, or 130 on interruption."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        camera_changes = {}
        if args.image is not None and args.camera_index is not None:
            raise ConfigurationError("Choose either --image or --camera-index for a run.")
        if args.image is not None:
            if args.source not in (None, "image"):
                raise ConfigurationError("--image requires --source image when a source is explicit.")
            camera_changes.update(source="image", image_path=args.image.expanduser().resolve())
        if args.camera_index is not None:
            if args.source not in (None, "webcam"):
                raise ConfigurationError("--camera-index requires --source webcam when a source is explicit.")
            camera_changes.update(source="webcam", webcam_index=args.camera_index)
        if args.source is not None:
            camera_changes["source"] = args.source
        simulation_changes = {}
        if args.max_frames is not None:
            simulation_changes["max_frames"] = args.max_frames
        if args.preview is not None:
            simulation_changes["preview"] = args.preview
        if args.output is not None:
            simulation_changes["output_path"] = args.output.expanduser().resolve()
        settings = replace(
            settings,
            camera=replace(settings.camera, **camera_changes),
            simulation=replace(settings.simulation, **simulation_changes),
            log_level=args.log_level or settings.log_level,
        )
        logging.basicConfig(level=settings.log_level, format="%(levelname)s: %(message)s")

        # Keep --help/--version available even if OpenCV has not been installed.
        from visionbridge.camera.capture import CaptureError
        from visionbridge.core.controller import SimulationError, run_simulation

        try:
            run_simulation(settings)
        except (CaptureError, SimulationError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1
    except (ConfigurationError, OSError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except ImportError as exc:
        logging.getLogger(__name__).error(
            "A local dependency is unavailable: %s. Install requirements.txt in your virtual environment.", exc
        )
        return 1
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Simulation stopped by the user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
