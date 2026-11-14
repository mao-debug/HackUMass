#!/usr/bin/env python3
"""List serial ports or explicitly sample local proximity, without camera/audio/API.

No port is opened by default or by --list. Sampling needs --port or --fixture.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

# Permit direct execution from an uninstalled checkout on Mac or Raspberry Pi.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="list discoverable ports without opening them")
    selection.add_argument("--port", help="explicit USB serial path; preserves by-id symlinks")
    selection.add_argument("--fixture", type=Path, help="sample an explicitly simulated JSON fixture")
    parser.add_argument("--samples", type=int, default=3, help="bounded number of sample requests (1..100)")
    args = parser.parse_args(argv)
    if not 1 <= args.samples <= 100:
        parser.error("--samples must be between 1 and 100")
    if args.list:
        try:
            from serial.tools import list_ports
            ports = [{"device": item.device, "description": item.description,
                      "serial_number": item.serial_number} for item in list_ports.comports()]
        except (ImportError, OSError):
            print(json.dumps({"status": "unavailable", "error": "Serial port listing requires optional pyserial support."}))
            return 1
        print(json.dumps({"status": "listed", "ports": ports}))
        return 0
    if not args.port and args.fixture is None:
        parser.print_help()
        return 0
    from vidi.core.configuration import ProximitySettings
    from vidi.sensors.proximity import ProximityError, create_source, unknown_snapshot
    settings = ProximitySettings(backend="serial" if args.port else "fixture",
                                 port=args.port, fixture_path=args.fixture)
    source = None
    try:
        source = create_source(settings)
        for _ in range(args.samples):
            print(json.dumps(asdict(source.sample()), sort_keys=True))
        return 0
    except ProximityError as exc:
        print(json.dumps(asdict(unknown_snapshot("unavailable", error=str(exc)))))
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "cancelled"}))
        return 130
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
