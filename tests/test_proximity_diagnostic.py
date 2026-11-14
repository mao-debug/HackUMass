"""The sensor diagnostic never opens an implicit port or invokes other hardware."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vidi.sensors import proximity


@pytest.fixture
def diagnostic():
    before = sys.path[:]
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_sensors.py"
    spec = importlib.util.spec_from_file_location("vidi_check_sensors_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = before


def test_no_arguments_prints_help_without_opening_any_source(diagnostic, monkeypatch, capsys):
    monkeypatch.setattr(proximity, "create_source", lambda _: pytest.fail("device opened"))
    assert diagnostic.main([]) == 0
    assert "--port" in capsys.readouterr().out


def test_list_only_discovers_ports_and_never_opens_them(diagnostic, monkeypatch, capsys):
    serial, tools = ModuleType("serial"), ModuleType("serial.tools")
    serial.__path__ = []
    serial.tools = tools
    tools.list_ports = SimpleNamespace(comports=lambda: [
        SimpleNamespace(device="/dev/ttyUSB0", description="USB bridge", serial_number="board-id"),
    ])
    monkeypatch.setitem(sys.modules, "serial", serial)
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setattr(proximity, "create_source", lambda _: pytest.fail("device opened"))
    assert diagnostic.main(["--list"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "listed" and result["ports"][0]["device"] == "/dev/ttyUSB0"


def test_fixture_diagnostic_is_finite_and_labels_simulation(diagnostic, tmp_path, capsys):
    fixture = tmp_path / "pair.json"
    fixture.write_text(json.dumps({"v": 1, "snapshots": [{
        "left": {"status": "ok", "distance_mm": 500},
        "right": {"status": "no_echo", "distance_mm": None},
    }]}))
    assert diagnostic.main(["--fixture", str(fixture), "--samples", "2"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2 and all(line["left"]["simulated"] for line in lines)
    assert lines[1]["right"]["status"] == "no_echo"


def test_explicit_port_failure_is_controlled_and_resources_closed(diagnostic, monkeypatch, capsys):
    closed = []

    class Failed:
        def sample(self):
            raise proximity.ProximityError("Protocol v1 firmware is required.")

        def close(self):
            closed.append(True)

    def factory(settings):
        assert settings.port == "/dev/serial/by-id/my-board"
        return Failed()

    monkeypatch.setattr(proximity, "create_source", factory)
    assert diagnostic.main(["--port", "/dev/serial/by-id/my-board", "--samples", "1"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["left"]["status"] == "unavailable" and "Protocol v1" in result["left"]["error"]
    assert closed == [True]


def test_ctrl_c_is_cancelled_and_closes_source(diagnostic, monkeypatch, capsys):
    closed = []
    source = SimpleNamespace(sample=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                             close=lambda: closed.append(True))
    monkeypatch.setattr(proximity, "create_source", lambda _: source)
    assert diagnostic.main(["--port", "explicit"]) == 130
    assert json.loads(capsys.readouterr().out)["status"] == "cancelled" and closed == [True]


@pytest.mark.parametrize("args", [["--samples", "0"], ["--samples", "101"], ["--list", "--port", "x"]])
def test_unbounded_or_ambiguous_diagnostic_requests_are_rejected(diagnostic, args):
    with pytest.raises(SystemExit) as caught:
        diagnostic.main(args)
    assert caught.value.code == 2
