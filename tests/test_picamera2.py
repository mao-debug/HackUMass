"""Ordinary Pi camera ownership and bounded transport without real hardware."""

import builtins
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vidi.camera import managed, picamera2
from vidi.camera.capture import CaptureError
from vidi.core.configuration import CameraSettings
from vidi.core.types import DetectionResult, FramePacket


def make_session(monkeypatch):
    image = np.full((60, 80, 3), (10, 20, 30), dtype=np.uint8)
    request = Mock()
    request.make_array.return_value = image
    request.release.side_effect = lambda: image.fill(255)
    camera = Mock()
    camera.capture_request.return_value = request
    camera.create_preview_configuration.return_value = {"main": {"format": "RGB888"}}
    factory = Mock(return_value=camera)
    monkeypatch.setattr(picamera2, "_load_picamera2", lambda: factory)
    settings = CameraSettings(source="picamera2", width=80, height=60, fps=10)
    session = picamera2._Picamera2Session(settings)
    return session, factory, camera, request, image


def test_generic_session_configures_camera_zero_and_returns_owned_bgr_without_inference(monkeypatch):
    session, factory, camera, request, image = make_session(monkeypatch)
    factory.assert_called_once_with(0)
    camera.create_preview_configuration.assert_called_once_with(
        main={"size": (80, 60), "format": "RGB888"}, controls={"FrameRate": 10},
        buffer_count=4, queue=False,
    )
    camera.configure.assert_called_once_with({"main": {"format": "RGB888"}})
    camera.start.assert_called_once_with(show_preview=False)
    packet = session.read_packet(12)
    assert packet.frame_id == 12
    assert packet.captured_monotonic_ns > 0
    assert packet.inference is None
    np.testing.assert_array_equal(packet.image_bgr[0, 0], (10, 20, 30))
    assert np.all(image == 255)
    assert not np.shares_memory(packet.image_bgr, image)
    camera.capture_request.assert_called_once_with(flush=True)
    request.release.assert_called_once_with()
    request.get_metadata.assert_not_called()
    session.close()
    session.close()
    camera.stop.assert_called_once_with()
    camera.close.assert_called_once_with()


def test_generic_loader_never_imports_imx500_or_camera_devices(monkeypatch):
    module = ModuleType("picamera2")
    factory = Mock()
    module.Picamera2 = factory
    monkeypatch.setitem(sys.modules, "picamera2", module)
    original_import = builtins.__import__

    def import_without_ai_camera(name, *args, **kwargs):
        if name.startswith("picamera2.devices") or "imx500" in name.lower():
            raise AssertionError("Ordinary camera adapter attempted to import an AI Camera helper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ai_camera)
    assert picamera2._load_picamera2() is factory


def test_missing_picamera_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2", None)
    with pytest.raises(CaptureError, match="Picamera2/libcamera"):
        picamera2._load_picamera2()


@pytest.mark.parametrize("image", [
    np.empty((0, 80, 3), dtype=np.uint8), np.zeros((60, 80), dtype=np.uint8),
    np.zeros((60, 80, 4), dtype=np.uint8), np.zeros((60, 80, 3), dtype=np.float32),
])
def test_invalid_images_release_the_request(monkeypatch, image):
    session, _, _, request, _ = make_session(monkeypatch)
    request.make_array.return_value = image
    with pytest.raises(CaptureError, match="nonempty uint8 BGR"):
        session.read_packet(1)
    request.release.assert_called_once_with()
    session.close()


def test_request_is_released_when_copying_image_fails(monkeypatch):
    session, _, _, request, _ = make_session(monkeypatch)
    request.make_array.side_effect = RuntimeError("camera buffer unavailable")
    with pytest.raises(RuntimeError, match="buffer unavailable"):
        session.read_packet(1)
    request.release.assert_called_once_with()
    session.close()


def test_no_completed_request_returns_none_without_inference(monkeypatch):
    session, _, camera, request, _ = make_session(monkeypatch)
    camera.capture_request.return_value = None
    assert session.read_packet(1) is None
    request.release.assert_not_called()
    session.close()


def test_setup_failure_closes_camera_and_preserves_primary_error(monkeypatch):
    camera = Mock()
    camera.configure.side_effect = RuntimeError("configuration failed")
    camera.stop.side_effect = RuntimeError("stop failed")
    camera.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(picamera2, "_load_picamera2", lambda: Mock(return_value=camera))
    with pytest.raises(RuntimeError, match="configuration failed"):
        picamera2._Picamera2Session(CameraSettings(source="picamera2"))
    camera.stop.assert_called_once_with()
    camera.close.assert_called_once_with()


def test_worker_builds_only_generic_session_and_transports_camera_only_packet(monkeypatch):
    packet = FramePacket(1, 12, np.full((60, 80, 3), 40, dtype=np.uint8))
    session = Mock()
    session.read_packet.return_value = packet
    factory = Mock(return_value=session)
    monkeypatch.setattr(picamera2, "_Picamera2Session", factory)
    frame_bytes = packet.image_bgr.nbytes
    memory = SimpleNamespace(buf=bytearray(frame_bytes + managed._RESULT_BYTES), close=Mock())
    monkeypatch.setattr(managed.shared_memory, "SharedMemory", Mock(return_value=memory))
    connection = Mock()
    connection.recv.side_effect = ["read", "close"]
    settings = CameraSettings(source="picamera2")
    picamera2._worker_main(connection, "fake-memory", frame_bytes, settings)
    factory.assert_called_once_with(settings)
    assert connection.send.call_args_list[0].args[0] == ("ready", 0)
    kind, length = connection.send.call_args_list[1].args[0]
    assert kind == "frame"
    data = json.loads(bytes(memory.buf[frame_bytes:frame_bytes + length]))
    assert data["frame_id"] == 1
    assert data["inference"] is None
    assert bytes(memory.buf[:frame_bytes]) == packet.image_bgr.tobytes()
    session.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def fake_open_source():
    source = picamera2.Picamera2Capture(
        CameraSettings(source="picamera2", width=80, height=60, capture_timeout_s=0.01)
    )
    source._opened = True
    source._connection = Mock()
    source._process = Mock(pid=123)
    source._process.is_alive.return_value = False
    source._memory = SimpleNamespace(
        buf=bytearray(source._frame_bytes + managed._RESULT_BYTES), close=Mock(), unlink=Mock()
    )
    return source


def test_generic_capture_timeout_uses_same_bounded_cleanup():
    source = fake_open_source()
    connection, process, memory = source._connection, source._process, source._memory
    connection.poll.return_value = False
    with pytest.raises(CaptureError, match="Picamera2 capture timed out"):
        source.read_packet()
    connection.poll.assert_called_once_with(0.01)
    process.join.assert_called_once_with(managed._JOIN_TIMEOUT_S)
    memory.close.assert_called_once_with()
    memory.unlink.assert_called_once_with()


def test_generic_source_rejects_unexpected_inference_in_worker_result():
    source = fake_open_source()
    packet = FramePacket(1, 12, np.zeros((60, 80, 3), dtype=np.uint8),
                         DetectionResult(1, "imx500", "ready"))
    encoded = managed._packet_metadata(packet)
    source._memory.buf[source._frame_bytes:source._frame_bytes + len(encoded)] = encoded
    source._connection.poll.return_value = True
    source._connection.recv.return_value = ("frame", len(encoded))
    with pytest.raises(CaptureError, match="unexpected inference"):
        source.read_packet()
    assert source._closed


def test_parent_opens_generic_worker_with_no_model_settings(monkeypatch):
    monkeypatch.setattr(managed.sys, "platform", "linux")
    settings = CameraSettings(source="picamera2", startup_timeout_s=0.1)
    source = picamera2.Picamera2Capture(settings)
    memory = SimpleNamespace(name="fake-memory", close=Mock(), unlink=Mock())
    monkeypatch.setattr(managed.shared_memory, "SharedMemory", Mock(return_value=memory))
    connection, child = Mock(), Mock()
    connection.poll.return_value = True
    connection.recv.return_value = ("ready", 0)
    process = Mock(pid=123)
    process.is_alive.return_value = False
    context = Mock()
    context.Pipe.return_value = connection, child
    context.Process.return_value = process
    monkeypatch.setattr(managed.multiprocessing, "get_context", Mock(return_value=context))
    with source:
        assert source._opened
    assert context.Process.call_args.kwargs["target"] is picamera2._worker_main
    assert context.Process.call_args.kwargs["args"] == (child, "fake-memory", source._frame_bytes, settings)
    connection.poll.assert_called_once_with(0.1)
    child.close.assert_called_once_with()
    memory.unlink.assert_called_once_with()
