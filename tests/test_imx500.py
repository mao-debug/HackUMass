"""AI Camera request ownership and process deadlines using fake hardware only."""

import json
import multiprocessing
from multiprocessing import shared_memory
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vidi.ai.object_detection import DetectionError
from vidi.camera import imx500
from vidi.camera.capture import CaptureError
from vidi.core.configuration import CameraSettings, DetectionSettings, ModelSettings
from vidi.core.types import Detection, DetectionResult, FramePacket


def hardware_session(monkeypatch, *, outputs=None, labels=None, enabled=True):
    image = np.full((60, 80, 3), (10, 20, 30), dtype=np.uint8)
    metadata = {"ScalerCrop": (0, 0, 4056, 3040), "CnnOutputTensor": [1.0]}
    request = Mock()
    request.make_array.return_value = image
    request.get_metadata.return_value = metadata
    request.released = False

    def release():
        request.released = True
        image[:] = 255  # Recycled driver buffers must not affect returned images.

    request.release.side_effect = release
    camera = Mock()
    camera.capture_request.return_value = request
    camera.create_preview_configuration.return_value = {"main": {"format": "RGB888"}}
    sensor = Mock()
    sensor.camera_num = 2
    sensor.network_intrinsics = SimpleNamespace(
        task="object detection", postprocess=None, labels=labels or ["person", "-", "", "chair"],
        bbox_order="yx", bbox_normalization=False, preserve_aspect_ratio=False,
    )
    sensor.get_input_size.return_value = (320, 320)
    sensor.get_outputs.return_value = outputs

    def map_coords(box, actual_metadata, actual_camera):
        assert actual_metadata is metadata
        assert actual_camera is camera
        assert request.released is False
        return (-2, 5, 52, 80)

    sensor.convert_inference_coords.side_effect = map_coords
    picamera_factory = Mock(return_value=camera)
    sensor_factory = Mock(return_value=sensor)
    monkeypatch.setattr(imx500, "_load_pi_dependencies", lambda: (picamera_factory, sensor_factory))
    session = imx500._PiSession(
        CameraSettings(source="imx500", width=80, height=60),
        ModelSettings(object_detection_backend="imx500" if enabled else "none"), DetectionSettings(),
    )
    return session, request, camera, sensor, metadata, image, sensor_factory


def valid_outputs():
    return [np.array([[[0.1, 0.1, 0.8, 0.8]]]), np.array([[0.9]]), np.array([[3.0]])]


def test_image_and_inference_use_same_request_before_release(monkeypatch):
    session, request, camera, sensor, metadata, image, _ = hardware_session(
        monkeypatch, outputs=valid_outputs()
    )
    packet = session.read_packet(9)
    assert packet.frame_id == packet.inference.frame_id == 9
    assert packet.inference.status == "ready"
    detection = packet.inference.detections[0]
    assert (detection.class_id, detection.label) == (3, "chair")
    assert detection.bbox_xyxy == (0, 5, 50, 60)
    np.testing.assert_array_equal(packet.image_bgr[0, 0], [10, 20, 30])
    assert np.all(image == 255)
    assert not np.shares_memory(packet.image_bgr, image)
    sensor.get_outputs.assert_called_once_with(metadata, add_batch=True)
    camera.capture_request.assert_called_once_with(flush=True)
    request.get_metadata.assert_called_once_with()
    request.release.assert_called_once_with()
    session.close()
    camera.stop.assert_called_once_with()
    camera.close.assert_called_once_with()


def test_missing_outputs_do_not_reuse_previous_detections(monkeypatch):
    session, request, _, sensor, _, _, _ = hardware_session(monkeypatch)
    sensor.get_outputs.side_effect = [valid_outputs(), None]
    ready = session.read_packet(1)
    request.released = False
    missing = session.read_packet(2)
    assert ready.inference.detections
    assert missing.inference.status == "not_ready"
    assert missing.inference.frame_id == 2
    assert missing.inference.detections == ()
    session.close()


def test_valid_empty_outputs_are_ready_not_warming_up(monkeypatch):
    outputs = [np.empty((1, 0, 4)), np.empty((1, 0)), np.empty((1, 0))]
    session, *_ = hardware_session(monkeypatch, outputs=outputs)
    result = session.read_packet(1).inference
    assert result.status == "ready"
    assert result.detections == ()
    session.close()


@pytest.mark.parametrize("category", [1, 2, 99, -1])
def test_undefined_or_unsupported_classes_are_skipped_without_renumbering(monkeypatch, category):
    outputs = valid_outputs()
    outputs[2] = np.array([[float(category)]])
    session, *_ = hardware_session(monkeypatch, outputs=outputs)
    assert session.read_packet(1).inference.detections == ()
    session.close()


@pytest.mark.parametrize("outputs", [
    [], [np.zeros((1, 4))],
    [np.zeros((1, 1, 3)), np.zeros((1, 1)), np.zeros((1, 1))],
    [np.zeros((1, 1, 4)), np.array([[float("nan")]]), np.zeros((1, 1))],
    [np.zeros((1, 1, 4)), np.array([[0.9]]), np.array([[0.2]])],
])
def test_incompatible_outputs_release_request_and_fail_clearly(monkeypatch, outputs):
    session, request, *_ = hardware_session(monkeypatch, outputs=outputs)
    with pytest.raises(DetectionError):
        session.read_packet(1)
    request.release.assert_called_once_with()
    session.close()


def test_request_is_released_when_coordinate_conversion_fails(monkeypatch):
    session, request, _, sensor, *_ = hardware_session(monkeypatch, outputs=valid_outputs())
    sensor.convert_inference_coords.side_effect = RuntimeError("bad crop")
    with pytest.raises(RuntimeError, match="bad crop"):
        session.read_packet(1)
    request.release.assert_called_once_with()
    session.close()


def test_camera_only_source_does_not_select_model_or_decode_outputs(monkeypatch):
    session, _, _, sensor, _, _, sensor_factory = hardware_session(monkeypatch, enabled=False)
    packet = session.read_packet(1)
    assert packet.inference is None
    sensor_factory.assert_called_once_with("")
    sensor.get_outputs.assert_not_called()
    sensor.get_input_size.assert_not_called()
    session.close()
    session.close()
    sensor.device_fd.close.assert_called_once_with()


def test_cleanup_failures_do_not_skip_descriptor_close_or_mask_primary_error(monkeypatch):
    session, _, camera, sensor, *_ = hardware_session(monkeypatch, outputs=valid_outputs())
    camera.stop.side_effect = RuntimeError("stop failed")
    camera.close.side_effect = RuntimeError("close failed")
    session.close()
    session.close()
    camera.stop.assert_called_once_with()
    camera.close.assert_called_once_with()
    sensor.device_fd.close.assert_called_once_with()


def fake_memory(size):
    return SimpleNamespace(name="fake-shared-memory", buf=bytearray(size), close=Mock(), unlink=Mock())


def active_capture(monkeypatch):
    source = imx500.Imx500Capture(
        CameraSettings(source="imx500", width=80, height=60, capture_timeout_s=0.01),
        ModelSettings(object_detection_backend="none"), DetectionSettings(),
    )
    source._opened = True
    source._connection = Mock()
    source._process = Mock(pid=123)
    source._process.is_alive.return_value = False
    source._memory = fake_memory(source._frame_bytes + imx500._RESULT_BYTES)
    return source


def test_capture_timeout_closes_ipc_process_and_shared_memory(monkeypatch):
    source = active_capture(monkeypatch)
    connection, process, memory = source._connection, source._process, source._memory
    connection.poll.return_value = False
    with pytest.raises(CaptureError, match="capture timed out"):
        source.read_packet()
    connection.poll.assert_called_once_with(0.01)
    connection.close.assert_called_once_with()
    process.join.assert_called_once_with(imx500._JOIN_TIMEOUT_S)
    memory.close.assert_called_once_with()
    memory.unlink.assert_called_once_with()
    source.close()
    memory.unlink.assert_called_once_with()


def test_blocked_worker_shutdown_escalates_with_bounded_joins(monkeypatch):
    source = active_capture(monkeypatch)
    process = source._process
    process.is_alive.side_effect = [True, True, False]
    source.close()
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.join.call_count == 3
    assert all(call.args == (imx500._JOIN_TIMEOUT_S,) for call in process.join.call_args_list)
    process.close.assert_called_once_with()


def test_startup_timeout_releases_every_resource(monkeypatch):
    monkeypatch.setattr(imx500.sys, "platform", "linux")
    source = imx500.Imx500Capture(
        CameraSettings(source="imx500", startup_timeout_s=0.02), ModelSettings(), DetectionSettings()
    )
    memory = fake_memory(source._frame_bytes + imx500._RESULT_BYTES)
    monkeypatch.setattr(imx500.shared_memory, "SharedMemory", Mock(return_value=memory))
    parent, child = Mock(), Mock()
    parent.poll.return_value = False
    process = Mock(pid=1)
    process.is_alive.return_value = False
    context = Mock()
    context.Pipe.return_value = parent, child
    context.Process.return_value = process
    monkeypatch.setattr(imx500.multiprocessing, "get_context", Mock(return_value=context))
    with pytest.raises(CaptureError, match="startup/model transfer timed out"):
        source.__enter__()
    parent.poll.assert_called_once_with(0.02)
    child.close.assert_called_once_with()
    parent.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    memory.unlink.assert_called_once_with()


def test_worker_error_response_becomes_controlled_detector_failure(monkeypatch):
    source = active_capture(monkeypatch)
    source._connection.poll.return_value = True
    source._connection.recv.return_value = ("detection_error", "unsupported model outputs")
    with pytest.raises(DetectionError, match="unsupported model outputs"):
        source.read_packet()
    assert source._closed


def test_repeated_frame_id_is_rejected_and_returned_image_owns_memory(monkeypatch):
    source = active_capture(monkeypatch)
    packet = FramePacket(1, 12, np.full((60, 80, 3), 40, dtype=np.uint8))
    encoded = imx500._packet_metadata(packet)
    memory = source._memory
    memory.buf[:packet.image_bgr.nbytes] = packet.image_bgr.tobytes()
    memory.buf[source._frame_bytes:source._frame_bytes + len(encoded)] = encoded
    source._connection.poll.return_value = True
    source._connection.recv.return_value = ("frame", len(encoded))
    result = source.read_packet()
    memory.buf[0] = 255
    assert result.image_bgr[0, 0, 0] == 40
    with pytest.raises(CaptureError, match="stale AI Camera frame"):
        source.read_packet()


def test_transport_refuses_inference_for_another_frame():
    packet = FramePacket(1, 1, np.zeros((60, 80, 3), dtype=np.uint8), DetectionResult(2, "imx500", "ready"))
    with pytest.raises(DetectionError, match="different camera frame"):
        imx500._packet_metadata(packet)


def test_worker_transports_owned_image_and_result_then_releases_resources(monkeypatch):
    packet = FramePacket(1, 12, np.full((60, 80, 3), 40, dtype=np.uint8),
                         DetectionResult(1, "imx500", "not_ready"))
    session = Mock()
    session.read_packet.return_value = packet
    monkeypatch.setattr(imx500, "_PiSession", Mock(return_value=session))
    frame_bytes = packet.image_bgr.nbytes
    memory = fake_memory(frame_bytes + imx500._RESULT_BYTES)
    monkeypatch.setattr(imx500.shared_memory, "SharedMemory", Mock(return_value=memory))
    connection = Mock()
    connection.recv.side_effect = ["read", "close"]
    imx500._worker_main(connection, "fake-memory", frame_bytes, None, None, None)
    assert connection.send.call_args_list[0].args == (("ready", 0),)
    kind, length = connection.send.call_args_list[1].args[0]
    assert kind == "frame"
    data = json.loads(bytes(memory.buf[frame_bytes:frame_bytes + length]))
    assert data["frame_id"] == 1
    assert data["inference"]["status"] == "not_ready"
    assert bytes(memory.buf[:frame_bytes]) == packet.image_bgr.tobytes()
    session.read_packet.assert_called_once_with(1)
    session.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    memory.unlink.assert_not_called()  # The parent, not worker, owns unlink.
    connection.close.assert_called_once_with()


def test_worker_setup_failure_reports_controlled_error_and_closes_transport(monkeypatch):
    memory = fake_memory(100)
    monkeypatch.setattr(imx500.shared_memory, "SharedMemory", Mock(return_value=memory))
    monkeypatch.setattr(imx500, "_PiSession", Mock(side_effect=CaptureError("Pi dependency missing")))
    connection = Mock()
    imx500._worker_main(connection, "fake-memory", 20, None, None, None)
    assert connection.send.call_args.args[0] == ("capture_error", "Pi dependency missing")
    memory.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_unsupported_model_is_rejected_without_starting_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(imx500.sys, "platform", "linux")
    model = tmp_path / "different-model.rpk"
    model.write_bytes(b"not a supported model")
    context = Mock()
    monkeypatch.setattr(imx500.multiprocessing, "get_context", context)
    source = imx500.Imx500Capture(
        CameraSettings(source="imx500"),
        ModelSettings(object_detection_backend="imx500", object_detection_path=model), DetectionSettings(),
    )
    with pytest.raises(DetectionError, match="other models need their own decoder"):
        source.read_packet()
    context.assert_not_called()


def test_non_linux_source_fails_before_creating_worker(monkeypatch):
    monkeypatch.setattr(imx500.sys, "platform", "darwin")
    context = Mock()
    monkeypatch.setattr(imx500.multiprocessing, "get_context", context)
    source = imx500.Imx500Capture(CameraSettings(source="imx500"), ModelSettings(), DetectionSettings())
    with pytest.raises(CaptureError, match="Raspberry Pi OS/Linux"):
        source.read_packet()
    context.assert_not_called()


def spawned_fake_camera_worker(connection, shared_name, frame_bytes):
    """Exercise real IPC with generated frames; never import or open Pi APIs."""
    memory = shared_memory.SharedMemory(name=shared_name)
    try:
        # Limit command waits and total frames even if the parent fails a test.
        for frame_id in range(1, 4):
            if not connection.poll(3.0):
                return
            command = connection.recv()
            if command == "close":
                return
            if command != "read":
                raise AssertionError(f"Unexpected fake-worker command: {command}")
            image = np.full((4, 6, 3), frame_id * 40, dtype=np.uint8)
            result = DetectionResult(
                frame_id, "imx500", "ready",
                (Detection(3, "chair", 0.9, (0, 0, 3, 4)),),
            )
            packet = FramePacket(frame_id, time.monotonic_ns(), image, result)
            encoded = imx500._packet_metadata(packet)
            memory.buf[:image.nbytes] = image.tobytes()
            memory.buf[frame_bytes:frame_bytes + len(encoded)] = encoded
            connection.send(("frame", len(encoded)))
    finally:
        memory.close()
        connection.close()


def test_spawned_fake_worker_round_trip_owns_frames_and_cleans_real_ipc():
    """Real spawn/Pipe/shared memory validate transport without camera hardware."""
    source = imx500.Imx500Capture(
        CameraSettings(source="imx500", width=6, height=4, capture_timeout_s=3.0),
        ModelSettings(object_detection_backend="none"), DetectionSettings(),
    )
    context = multiprocessing.get_context("spawn")
    try:
        memory = shared_memory.SharedMemory(create=True, size=source._frame_bytes + imx500._RESULT_BYTES)
    except PermissionError:
        pytest.skip("OS sandbox blocks POSIX shared memory; real spawned IPC smoke was not executed.")
    shared_name = memory.name
    source._memory = memory
    child = None
    try:
        parent, child = context.Pipe()
        source._connection = parent
        process = context.Process(
            target=spawned_fake_camera_worker, args=(child, shared_name, source._frame_bytes), daemon=True,
        )
        # Inject an already-open transport so production _open never selects a
        # platform, imports Picamera2, or creates a camera. Only read/close are real.
        source._process = process
        source._opened = True
        process.start()
        child.close()
        first, second = source.read_packet(), source.read_packet()
        assert (first.frame_id, second.frame_id) == (1, 2)
        assert first.captured_monotonic_ns < second.captured_monotonic_ns
        assert first.inference.frame_id == first.frame_id
        assert second.inference.frame_id == second.frame_id
        assert second.inference.detections[0].bbox_xyxy == (0, 0, 3, 4)
        assert np.all(first.image_bgr == 40)
        assert np.all(second.image_bgr == 80)
        assert not np.shares_memory(first.image_bgr, second.image_bgr)
    finally:
        if child is not None:
            child.close()
        source.close()
    assert process._closed  # close() joined the real child and released its handle.
    assert parent.closed
    assert np.all(first.image_bgr == 40)
    assert np.all(second.image_bgr == 80)
    with pytest.raises(CaptureError, match="source is closed"):
        source.read_packet()
    try:
        leaked_memory = shared_memory.SharedMemory(name=shared_name)
    except FileNotFoundError:
        pass
    else:
        leaked_memory.close()
        pytest.fail("Camera shared memory was not unlinked during close.")
