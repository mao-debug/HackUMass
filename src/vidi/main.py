"""Local capture/detection CLI with an explicit optional Gemini request."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from vidi import __version__
from vidi.core.configuration import ConfigurationError, load_settings, validate_runtime_settings


def build_parser() -> argparse.ArgumentParser:
    """Describe local backends and the per-run cloud opt-in."""
    parser = argparse.ArgumentParser(
        description="vidi Phase 2: local AI Camera detection, simulation, and optional Gemini descriptions."
    )
    parser.add_argument("--version", action="version", version=f"vidi {__version__}")
    parser.add_argument("--config", type=Path, help="YAML path; omitted means built-in defaults")
    parser.add_argument("--source", choices=("sample", "image", "webcam", "imx500"))
    parser.add_argument("--image", type=Path, help="Image path (selects image unless --source is given)")
    parser.add_argument("--camera-index", type=int, help="Webcam index (selects webcam unless --source is given)")
    parser.add_argument("--max-frames", type=int, help="Positive frame limit (default: 1)")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None,
                        help="Open a local preview window; requires a desktop display")
    parser.add_argument("--output", type=Path, help="Save the last frame to a local image path")
    parser.add_argument("--detector", choices=("none", "imx500", "fixture"), help="Explicit detector selection")
    parser.add_argument("--model", type=Path, help="Local IMX500 RPK file; never downloaded automatically")
    parser.add_argument("--labels", type=Path, help="Model-aligned label file, preserving class ID gaps")
    parser.add_argument("--threshold", type=float, help="Confidence threshold between 0 and 1")
    parser.add_argument("--max-detections", type=int, help="Positive maximum detections per frame")
    parser.add_argument("--detections-fixture", type=Path, help="Explicit simulated detection data")
    parser.add_argument("--report", type=Path, help="Write local detection and timing results as JSON")
    parser.add_argument("--describe", action="store_true", help="Send ONE last unannotated image to Gemini (cloud opt-in)")
    parser.add_argument("--question", help="Optional typed question for --describe; does not enable voice input")
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
        if args.report is not None:
            simulation_changes["report_path"] = args.report.expanduser().resolve()
        model_changes = {}
        if args.detector is not None:
            model_changes["object_detection_backend"] = args.detector
        if args.model is not None:
            model_changes["object_detection_path"] = args.model.expanduser().resolve()
        detection_changes = {}
        if args.labels is not None:
            detection_changes["labels_path"] = args.labels.expanduser().resolve()
        if args.detections_fixture is not None:
            detection_changes["fixture_path"] = args.detections_fixture.expanduser().resolve()
        if args.threshold is not None:
            detection_changes["confidence_threshold"] = args.threshold
        if args.max_detections is not None:
            detection_changes["max_detections"] = args.max_detections
        if args.question is not None and not args.describe:
            raise ConfigurationError("--question requires --describe (an explicit Gemini upload opt-in).")
        if args.question is not None and (not args.question.strip() or len(args.question) > 500):
            raise ConfigurationError("--question must contain between 1 and 500 characters.")
        settings = replace(
            settings,
            camera=replace(settings.camera, **camera_changes),
            simulation=replace(settings.simulation, **simulation_changes),
            models=replace(settings.models, **model_changes),
            detection=replace(settings.detection, **detection_changes),
            log_level=args.log_level or settings.log_level,
        )
        logging.basicConfig(level=settings.log_level, format="%(levelname)s: %(message)s")
        validate_runtime_settings(settings, describe=args.describe)

        # Keep --help/--version available even if OpenCV has not been installed.
        from vidi.camera.capture import CaptureError
        from vidi.ai.object_detection import DetectionError
        from vidi.core.controller import SimulationError, run_simulation

        try:
            if args.describe:
                result = run_simulation(settings, describe=True, question=args.question or "What is in front of me?")
                if result.description_status != "ready":
                    return 1
            else:
                result = run_simulation(settings)
            if (settings.models.object_detection_backend or "none") != "none" and result.ready_frames == 0:
                logging.getLogger(__name__).error(
                    "No inference results arrived during this run. Check model startup and camera diagnostics; "
                    "increase --max-frames for warm-up."
                )
                return 1
        except (CaptureError, DetectionError, SimulationError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1
    except (ConfigurationError, OSError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except ImportError as exc:
        logging.getLogger(__name__).error(
            "A local dependency is unavailable: %s. See README.md for desktop requirements or the Pi OS package setup.", exc
        )
        return 1
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Simulation stopped by the user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
