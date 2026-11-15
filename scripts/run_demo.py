#!/usr/bin/env python3
"""Plan or launch vidi with an existing environment; simulation is offline by default.

The default invocation only prints a command. --run hands the foreground process
to vidi, so its own cancellation and exit-status handling remain in control.
No packages, models, credentials, or audio routes are installed or changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LaunchPlan:
    """Explicit application arguments without inherited environment or secrets."""

    argv: tuple[str, ...]
    root: Path
    report_path: Path
    target: str
    cloud: bool
    voice: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--target", choices=("pi", "sim"), default="sim",
                        help="Pi hardware or synthetic Mac simulation (default: sim)")
    parser.add_argument("--run", action="store_true", help="launch the displayed plan; otherwise nothing starts")
    parser.add_argument("--cloud", action="store_true", help="enable requested Gemini actions; otherwise uploads are disabled")
    parser.add_argument("--voice", action="store_true", help="enable M-triggered Gemini command clips; requires --cloud")
    parser.add_argument("--speak", action="store_true", help="speak requested responses with the existing local engine")
    parser.add_argument("--proximity", action="store_true", help="enable explicit Pi sensing or simulated proximity fixtures")
    parser.add_argument("--sensor-port", help="literal Pi USB serial path; requires --target pi --proximity")
    parser.add_argument("--command-audio", type=Path, help="explicit replay WAV instead of recording; requires --voice")
    parser.add_argument("--microphone-device", help="literal Pi ALSA input device; requires --target pi --voice")
    parser.add_argument("--prompt-key", action="store_true", help="use vidi's hidden per-run key prompt; requires --cloud")
    parser.add_argument("--python", type=Path, dest="python_path", help="override the existing target virtual-environment interpreter")
    parser.add_argument("--artifacts-dir", type=Path, help="report directory (default: repository artifacts/)")
    return parser


def _absolute(path: Path) -> Path:
    # Resolving a venv interpreter symlink would bypass its pyvenv.cfg.
    return Path(os.path.abspath(path.expanduser()))


def build_plan(args: argparse.Namespace, root: Path = PROJECT_ROOT) -> LaunchPlan:
    """Build a plan without importing vidi, reading a key, or opening devices."""
    if args.voice and not args.cloud:
        raise ValueError("--voice requires --cloud because command transcription sends audio to Gemini.")
    if args.prompt_key and not args.cloud:
        raise ValueError("--prompt-key requires --cloud; offline rehearsal does not load credentials.")
    if args.sensor_port is not None and (args.target != "pi" or not args.proximity):
        raise ValueError("--sensor-port requires --target pi --proximity.")
    if args.target == "pi" and args.proximity and not args.sensor_port:
        raise ValueError("Pi proximity requires an explicit --sensor-port from scripts/check_sensors.py --list.")
    if args.command_audio is not None and not args.voice:
        raise ValueError("--command-audio requires --voice.")
    if args.microphone_device is not None and (args.target != "pi" or not args.voice):
        raise ValueError("--microphone-device requires --target pi --voice.")
    if args.microphone_device is not None and args.command_audio is not None:
        raise ValueError("Choose --command-audio or --microphone-device, not both.")
    if args.target == "sim" and args.voice and args.command_audio is None:
        raise ValueError("Simulation voice needs --command-audio FILE.wav; it does not record a microphone.")
    for name, value in (("--microphone-device", args.microphone_device), ("--sensor-port", args.sensor_port)):
        if value is not None and (not value.strip() or any(c in value for c in "\x00\r\n")):
            raise ValueError(f"{name} must be a nonempty literal device name without NUL or newlines.")
    if args.command_audio is not None and args.command_audio.suffix.lower() != ".wav":
        raise ValueError("--command-audio requires a local .wav file of at most five seconds.")

    root = root.resolve()
    environment = ".venv-pi" if args.target == "pi" else ".venv"
    python = _absolute(args.python_path if args.python_path is not None else root / environment / "bin/python")
    profile = root / "config" / ("pi-phase5.yaml" if args.target == "pi" else "phase5-sim.yaml")
    directory = _absolute(args.artifacts_dir) if args.artifacts_dir is not None else root / "artifacts"
    report = directory / f"demo-{args.target}-{uuid.uuid4().hex[:12]}.json"
    command = [str(python), "-m", "vidi", "--config", str(profile),
               "--continuous", "--scene-live", "--preview", "--report", str(report)]
    if not args.cloud:
        command.append("--offline")
    if args.voice:
        command.append("--voice-commands")
    if args.speak:
        command.append("--speak")
    if args.proximity:
        command.append("--proximity")
    if args.sensor_port is not None:
        command.extend(("--sensor-port", args.sensor_port))
    if args.command_audio is not None:
        command.extend(("--command-audio", str(_absolute(args.command_audio))))
    if args.microphone_device is not None:
        command.extend(("--microphone-device", args.microphone_device))
    if args.prompt_key:
        command.append("--prompt-key")
    return LaunchPlan(tuple(command), root, report, args.target, args.cloud, args.voice)


def _exec_plan(plan: LaunchPlan) -> None:
    interpreter = Path(plan.argv[0])
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise OSError("The selected Python interpreter is unavailable.")
    plan.report_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE="1", PYTHONPATH=str(plan.root / "src"))
    os.chdir(plan.root)
    # Replace the wrapper: the application receives Ctrl+C directly and owns
    # cleanup. Passing an argument list avoids shell interpretation of paths.
    os.execve(plan.argv[0], list(plan.argv), environment)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as error:
        parser.error(str(error))
    print("vidi demo: " + ("Pi hardware" if plan.target == "pi" else "synthetic simulation"))
    print("Gemini actions: " + ("enabled on request" if plan.cloud else "disabled (offline)"))
    if plan.voice:
        print("M sends one requested command clip; a recognized camera command can then send one image.")
    elif plan.cloud:
        print("D requests one scene image. Add --voice for M commands and T text reading.")
    else:
        print("D/T/M are disabled. H is local help; P reads proximity only when --proximity is selected.")
    print("Report: " + str(plan.report_path))
    print("Command: PYTHONNOUSERSITE=1 PYTHONPATH=" + shlex.quote(str(plan.root / "src"))
          + " " + shlex.join(plan.argv), flush=True)
    if not args.run:
        print("Plan only. Add --run to start the application.")
        return 0
    try:
        _exec_plan(plan)
    except OSError:
        print("Launch failed. Verify the existing target environment or select --python PATH. No installation was attempted.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
