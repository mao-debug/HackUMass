"""Local detection and reading CLI with explicit speech and cloud requests."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path

from vidi import __version__
from vidi.core.configuration import ConfigurationError, load_settings, validate_runtime_settings
from vidi.core.credentials import CredentialError, prompted_gemini_key, save_gemini_key, saved_gemini_key


def build_parser() -> argparse.ArgumentParser:
    """Describe local backends and the per-run cloud opt-in."""
    parser = argparse.ArgumentParser(
        description="vidi Phase 5: requested camera scenes, local proximity, printed-text reading, and user-triggered voice commands.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"vidi {__version__}")
    parser.add_argument("--config", type=Path, help="YAML path; omitted means built-in defaults")
    parser.add_argument("--source", choices=("sample", "image", "webcam", "picamera2", "imx500"))
    parser.add_argument("--image", type=Path, help="Image path (selects image unless --source is given)")
    parser.add_argument("--camera-index", type=int, help="Webcam index (selects webcam unless --source is given)")
    parser.add_argument("--max-frames", type=int, help="Positive frame limit (default: 1)")
    parser.add_argument("--continuous", action="store_true", help="Keep the camera running until q/Esc or Ctrl+C; cloud actions require --gemini-live or --scene-live")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None,
                        help="Open a local preview window; requires a desktop display")
    parser.add_argument("--output", type=Path, help="Save the last frame to a local image path")
    parser.add_argument("--detector", choices=("none", "imx500", "opencv_ssd", "fixture", "gemini"), help="Explicit detector selection")
    parser.add_argument("--model", type=Path, help="Local IMX500 RPK or MobileNet-SSD Caffe weights; no automatic download")
    parser.add_argument("--model-config", type=Path, help="Local MobileNet-SSD deploy.prototxt for opencv_ssd")
    parser.add_argument("--labels", type=Path, help="Model-aligned label file, preserving class ID gaps")
    parser.add_argument("--threshold", type=float, help="Confidence threshold between 0 and 1")
    parser.add_argument("--max-detections", type=int, help="Positive maximum detections per frame")
    parser.add_argument("--detections-fixture", type=Path, help="Explicit simulated detection data")
    parser.add_argument("--report", type=Path, help="Write local detection and timing results as JSON")
    parser.add_argument("--read-text", action="store_true", help="Read printed text once from the final raw frame (local OCR)")
    parser.add_argument("--gemini-read", action="store_true", help="Send ONE final raw image to Gemini to transcribe printed text (cloud opt-in)")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--speak", action="store_true", help="Speak requested reading, scene, or local proximity feedback")
    speech.add_argument("--speech-output", type=Path, help="Synthesize one requested reading or Gemini result to WAV without playback")
    parser.add_argument("--gemini-detect", action="store_true", help="Send ONE final raw image to Gemini for object labels and boxes (cloud opt-in)")
    parser.add_argument("--gemini-live", action="store_true", help="With --continuous, repeatedly send frames to Gemini, no faster than once per 5 seconds; stop uploads on error")
    parser.add_argument("--prompt-key", action="store_true", help="Enter the Gemini key privately for this run; accepts no key argument")
    parser.add_argument("--save-key", action="store_true", help="Privately enter and save a Gemini key locally, then exit; accepts no key argument")
    parser.add_argument("--describe", action="store_true", help="Send ONE last unannotated image to Gemini (cloud opt-in)")
    parser.add_argument("--scene-live", action="store_true", help="With --continuous --preview, D requests one scene image, P reads local proximity; no automatic uploads or speech")
    parser.add_argument("--offline", action="store_true", help="With --scene-live --continuous --preview, disable cloud actions and microphone; P/H remain local")
    parser.add_argument("--voice-commands", action="store_true", help="With --scene-live --continuous --preview, M records up to 5 seconds and sends audio to Gemini; no continuous listening")
    parser.add_argument("--command-audio", type=Path, help="On M, send this local WAV instead of recording a microphone; requires --voice-commands")
    parser.add_argument("--microphone-device", help="Literal ALSA input-device name overriding voice.device; requires --voice-commands")
    parser.add_argument("--proximity", action="store_true", help="Explicitly enable local sensor readings with --describe or --scene-live")
    parser.add_argument("--sensor-port", help="Explicit ESP32 USB serial port; requires --proximity (no automatic port selection)")
    parser.add_argument("--question", help="Optional typed question for --describe or --scene-live; does not enable voice input")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return 0 on success, 1 on an expected failure, or 130 on interruption."""
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    for flag in ("--prompt-key", "--save-key"):
        if any(argument.startswith(f"{flag}=") for argument in arguments):
            parser.error(f"{flag} takes no key argument. Enter the raw key only at the hidden prompt.")
    args, unknown = parser.parse_known_args(arguments)
    if unknown:
        if args.prompt_key or args.save_key:
            parser.error("Key flags take no key argument. Enter the raw key only at the hidden prompt.")
        parser.error("Unrecognized arguments. Check --help for supported options.")
    try:
        if args.save_key:
            if arguments != ["--save-key"]:
                raise ConfigurationError("Use --save-key by itself to save the key without capturing or uploading an image.")
            save_gemini_key()
            print("Gemini key saved locally. Omit --prompt-key on future Gemini runs.")
            return 0
        if args.continuous and args.max_frames is not None:
            raise ConfigurationError("Choose --continuous or --max-frames; continuous mode runs until stopped.")
        if args.offline and (args.prompt_key or args.voice_commands):
            raise ConfigurationError("--offline cannot be combined with --prompt-key or --voice-commands.")
        settings = load_settings(args.config)
        if (args.command_audio is not None or args.microphone_device is not None) and not args.voice_commands:
            raise ConfigurationError("--command-audio and --microphone-device require --voice-commands.")
        if args.command_audio is not None and args.microphone_device is not None:
            raise ConfigurationError("Choose --command-audio or --microphone-device for voice input, not both.")
        if args.microphone_device is not None and settings.voice.audio_path is not None:
            raise ConfigurationError("Remove voice.audio_path before selecting --microphone-device.")
        voice_changes = {}
        if args.command_audio is not None:
            voice_changes["audio_path"] = args.command_audio.expanduser().resolve()
        if args.microphone_device is not None:
            voice_changes["device"] = args.microphone_device
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
        if args.gemini_detect or args.gemini_live:
            if args.detector not in (None, "gemini"):
                raise ConfigurationError("The Gemini action conflicts with the explicitly selected local detector.")
            model_changes["object_detection_backend"] = "gemini"
        if args.gemini_read:
            model_changes["ocr_backend"] = "gemini"
        if args.model is not None:
            model_changes["object_detection_path"] = args.model.expanduser().resolve()
        if args.model_config is not None:
            model_changes["object_detection_config_path"] = args.model_config.expanduser().resolve()
        detection_changes = {}
        if args.labels is not None:
            detection_changes["labels_path"] = args.labels.expanduser().resolve()
        if args.detections_fixture is not None:
            detection_changes["fixture_path"] = args.detections_fixture.expanduser().resolve()
        if args.threshold is not None:
            detection_changes["confidence_threshold"] = args.threshold
        if args.max_detections is not None:
            detection_changes["max_detections"] = args.max_detections
        if args.question is not None and not (args.describe or args.scene_live):
            raise ConfigurationError("--question requires --describe or --scene-live (explicit Gemini upload opt-ins).")
        if args.question is not None and (not args.question.strip() or len(args.question) > 500):
            raise ConfigurationError("--question must contain between 1 and 500 characters.")
        settings = replace(
            settings,
            camera=replace(settings.camera, **camera_changes),
            simulation=replace(settings.simulation, **simulation_changes),
            models=replace(settings.models, **model_changes),
            detection=replace(settings.detection, **detection_changes),
            log_level=args.log_level or settings.log_level,
            proximity=replace(settings.proximity, port=args.sensor_port) if args.sensor_port is not None else settings.proximity,
            voice=replace(settings.voice, **voice_changes),
        )
        if args.sensor_port is not None and not args.proximity:
            raise ConfigurationError("--sensor-port requires --proximity; configuration alone never opens a sensor.")
        if settings.models.object_detection_backend == "gemini":
            if args.threshold is not None:
                raise ConfigurationError("--threshold applies to scored local detectors; Gemini returns no confidence scores.")
            if args.model is not None or args.model_config is not None:
                raise ConfigurationError("Gemini uses cloud.model; --model and --model-config select local model files.")
        logging.basicConfig(level=settings.log_level, format="%(levelname)s: %(message)s")
        speech_output = args.speech_output.expanduser().resolve() if args.speech_output is not None else None
        voice_options = {"voice_commands": True} if args.voice_commands else {}
        offline_options = {"offline": True} if args.offline else {}
        validate_runtime_settings(
            settings, describe=args.describe, read_text=args.read_text,
            speak=args.speak, speech_output=speech_output, gemini_detect=args.gemini_detect,
            continuous=args.continuous,
            gemini_live=args.gemini_live,
            gemini_read=args.gemini_read,
            scene_live=args.scene_live, proximity=args.proximity,
            **voice_options,
            **offline_options,
        )
        if args.prompt_key and not (args.gemini_detect or args.describe or args.gemini_live or args.gemini_read or args.scene_live):
            raise ConfigurationError("--prompt-key requires --gemini-detect, --gemini-live, --gemini-read, --describe, or --scene-live.")

        # Keep --help/--version available even if OpenCV has not been installed.
        from vidi.camera.capture import CaptureError
        from vidi.ai.object_detection import DetectionError
        from vidi.core.controller import SimulationError, run_simulation

        key_context = (
            nullcontext() if args.offline
            else prompted_gemini_key() if args.prompt_key
            else saved_gemini_key() if args.gemini_detect or args.describe or args.gemini_live or args.gemini_read or args.scene_live
            else nullcontext()
        )
        with ExitStack() as contexts:
            try:
                contexts.enter_context(key_context)
            except CredentialError as exc:
                if not args.scene_live or args.prompt_key:
                    raise
                logging.getLogger(__name__).warning(
                    "Saved scene credentials unavailable: %s. Local proximity remains available.", exc,
                )
            try:
                if args.scene_live:
                    result = run_simulation(
                        settings, continuous=True, scene_live=True, proximity=args.proximity,
                        speak=args.speak, question=args.question or "What is in front of me?",
                        **voice_options,
                        **offline_options,
                    )
                    return 130 if result.interrupted else 0
                elif args.gemini_live:
                    result = run_simulation(settings, continuous=True, gemini_live=True)
                    return 130 if result.interrupted else (1 if result.last_inference is not None and result.last_inference.status != "ready" else 0)
                elif args.gemini_detect:
                    result = run_simulation(settings, gemini_detect=True, speak=args.speak, speech_output=speech_output)
                    if result.last_inference is None or result.last_inference.status != "ready" or result.speech.status == "failed":
                        return 1
                elif args.read_text or args.gemini_read:
                    reading_options = {"gemini_read": True} if args.gemini_read else {"read_text": True}
                    result = run_simulation(settings, **reading_options, speak=args.speak, speech_output=speech_output)
                    if result.ocr.status == "failed" or result.speech.status == "failed":
                        return 1
                elif args.describe:
                    options = {"describe": True, "question": args.question or "What is in front of me?"}
                    if args.proximity:
                        options["proximity"] = True
                    if args.speak or speech_output is not None:
                        options.update(speak=args.speak, speech_output=speech_output)
                    result = run_simulation(settings, **options)
                    if result.description_status != "ready" or result.speech.status == "failed":
                        return 1
                else:
                    result = run_simulation(settings, continuous=True) if args.continuous else run_simulation(settings)
                    if args.continuous and result.interrupted:
                        return 130
                if (settings.models.object_detection_backend or "none") != "none" and result.ready_frames == 0:
                    logging.getLogger(__name__).error(
                        "No inference results arrived during this run. Check model startup and camera diagnostics; "
                        "increase --max-frames for warm-up."
                    )
                    return 1
            except (CaptureError, DetectionError, SimulationError) as exc:
                logging.getLogger(__name__).error("%s", exc)
                return 1
    except (ConfigurationError, CredentialError, OSError) as exc:
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
