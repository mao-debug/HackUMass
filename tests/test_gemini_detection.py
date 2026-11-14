"""Gemini object detection tests: fake credentials and no external connections."""

import base64
import builtins
import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.ai import gemini_detection as gemini
from vidi.ai import scene_description as scene
from vidi.ai.gemini_detection import GeminiObjectDetector
from vidi.core.configuration import CloudSettings, DetectionSettings
from vidi.core.types import FramePacket


TEST_KEY = "fake-gemini-detection-key-not-a-real-credential"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", TEST_KEY)

    def deny_network(*args, **kwargs):
        pytest.fail("Gemini detection tests must never open an external connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)


def packet(frame_id=7, *, width=301, height=153):
    return FramePacket(frame_id, 123, np.full((height, width, 3), (20, 80, 180), np.uint8))


def document():
    return {"objects": [{"label": "person", "box_2d": [100, 200, 900, 800]}]}


def fake_client(*, text=None, response=None, error=None):
    if response is None:
        response = SimpleNamespace(
            status="completed", output_text=json.dumps(document()) if text is None else text,
        )
    client = SimpleNamespace(
        interactions=SimpleNamespace(create=Mock(return_value=response, side_effect=error)), close=Mock(),
    )
    return client, Mock(return_value=client)


def test_constructor_does_not_read_credentials_or_create_a_client(monkeypatch):
    environment = SimpleNamespace(get=Mock(side_effect=AssertionError("Credential access too early")))
    monkeypatch.setattr(gemini, "os", SimpleNamespace(environ=environment))
    factory = Mock(side_effect=AssertionError("Client creation too early"))

    GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory)

    environment.get.assert_not_called()
    factory.assert_not_called()


def test_ready_boxes_have_no_invented_confidence_or_class_id(monkeypatch):
    client, factory = fake_client()
    cloud = CloudSettings()
    monkeypatch.setattr(gemini, "time", SimpleNamespace(monotonic=Mock(side_effect=[10.0, 10.75])))

    result = GeminiObjectDetector(cloud, DetectionSettings(), client_factory=factory).detect(packet(42))

    assert result.frame_id == 42 and result.backend == "gemini" and result.status == "ready"
    assert result.model == cloud.model and result.elapsed_s == 0.75
    assert result.simulated is False and result.error is None
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.label == "person"
    assert detection.class_id is None and detection.confidence is None
    assert detection.bbox_xyxy == (60, 15, 241, 138)
    client.close.assert_called_once_with()


def test_request_has_strict_schema_and_explicit_single_stateless_request():
    client, factory = fake_client()
    cloud = CloudSettings(timeout_s=4.25, detection_max_output_tokens=768)
    settings = DetectionSettings(max_detections=3, confidence_threshold=1)

    result = GeminiObjectDetector(cloud, settings, client_factory=factory).detect(packet())

    assert result.status == "ready"  # No confidence filter can apply to absent scores.
    factory.assert_called_once_with(
        api_key=TEST_KEY, http_options={"timeout": 4250, "retry_options": {"attempts": 0}},
    )
    request = client.interactions.create.call_args.kwargs
    assert request["model"] == cloud.model
    assert request["generation_config"] == {"max_output_tokens": 768, "tool_choice": "none"}
    assert cloud.max_output_tokens == 256  # Scene-description default stays independent.
    assert request["response_format"]["type"] == "text"
    assert request["response_format"]["mime_type"] == "application/json"
    schema = request["response_format"]["schema"]
    assert schema["additionalProperties"] is False and schema["required"] == ["objects"]
    objects = schema["properties"]["objects"]
    assert objects["maxItems"] == 3
    item = objects["items"]
    assert item["additionalProperties"] is False
    assert set(item["properties"]) == {"label", "box_2d"}
    assert set(item["required"]) == {"label", "box_2d"}
    assert item["properties"]["box_2d"] == {
        "type": "array", "minItems": 4, "maxItems": 4,
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    }
    for name in ("store", "background", "stream", "automatic_continuation"):
        assert request[name] is False
    assert request["tools"] == [] and request["timeout"] == 4.25
    assert "previous_interaction_id" not in request
    assert request["input"][1]["mime_type"] == "image/jpeg"
    assert "uri" not in request["input"][1]
    for phrase in ("English", "[ymin, xmin, ymax, xmax]", "confidence scores", "class IDs", "masks",
                   "sensitive personal attributes", "precise distances", "navigation instructions",
                   "untrusted observation", "Do not use tools"):
        assert phrase in request["system_instruction"]


def test_resize_upload_maps_boxes_to_original_image_and_never_mutates_it():
    frame = packet(width=2000, height=1200)
    original = frame.image_bgr.copy()
    payload = {"objects": [{"label": "chair", "box_2d": [125, 250, 875, 750]}]}
    client, factory = fake_client(text=json.dumps(payload))

    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(frame)

    data = base64.b64decode(client.interactions.create.call_args.kwargs["input"][1]["data"], validate=True)
    uploaded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert uploaded.shape[:2] == (614, 1024)
    assert len(data) <= scene.MAX_IMAGE_BYTES
    np.testing.assert_allclose(uploaded[100, 100], original[100, 100], atol=4)
    np.testing.assert_array_equal(frame.image_bgr, original)
    assert result.detections[0].bbox_xyxy == (500, 150, 1500, 1050)


@pytest.mark.parametrize("box,expected", [
    ([0, 0, 1000, 1000], (0, 0, 301, 153)),
    ([1, 1, 2, 2], (0, 0, 1, 1)),
])
def test_pixel_floor_and_ceil_include_edges_and_subpixel_objects(box, expected):
    _, factory = fake_client(text=json.dumps({"objects": [{"label": "chair", "box_2d": box}]}))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "ready" and result.detections[0].bbox_xyxy == expected


def test_whitespace_around_a_valid_label_is_trimmed():
    payload = document()
    payload["objects"][0]["label"] = "  potted plant  "
    _, factory = fake_client(text=json.dumps(payload))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.detections[0].label == "potted plant"


def test_empty_objects_is_a_ready_empty_result():
    _, factory = fake_client(text='{"objects": []}')
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "ready" and result.detections == () and result.error is None


@pytest.mark.parametrize("text", [
    "not JSON", "```json\n{\"objects\": []}\n```", "null", "[]", "{}",
    '{"objects": null}', '{"objects": {}}', '{"objects": [], "extra": 1}',
    '{"objects": [], "objects": []}',
    '{"objects": [{"label": "chair", "label": "person", "box_2d": [0,0,1000,1000]}]}',
    '{"objects": []}' + " " * 65537,
])
def test_invalid_or_ambiguous_json_is_unavailable_without_exposing_payload(text, caplog):
    client, factory = fake_client(text=text)
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()
    assert result.error == "Gemini returned invalid object detection data. Try another snapshot."
    assert caplog.text == ""
    client.close.assert_called_once_with()


@pytest.mark.parametrize("entry", [
    None, [], {}, {"label": "chair"}, {"box_2d": [0, 0, 1000, 1000]},
    {"label": "chair", "box_2d": [0, 0, 1000, 1000], "confidence": 0.9},
    {"label": "chair", "box_2d": [0, 0, 1000, 1000], "class_id": 9},
    {"label": "chair", "box_2d": [0, 0, 1000, 1000], "mask": []},
])
def test_incorrect_object_fields_are_rejected(entry):
    _, factory = fake_client(text=json.dumps({"objects": [entry]}))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()


@pytest.mark.parametrize("label", [
    None, True, 42, [], "", "   ", "x" * 65, "chair\n", "chair\x00", "chair\u202e",
    "-", "???", "N/A", "Unknown", "background", "__background__", "null", "none",
    "object " + TEST_KEY, TEST_KEY.upper(),
])
def test_invalid_placeholder_or_credential_labels_are_never_returned(label, caplog):
    payload = document()
    payload["objects"][0]["label"] = label
    _, factory = fake_client(text=json.dumps(payload))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()
    assert TEST_KEY not in result.error and "chair" not in result.error
    assert caplog.text == ""


@pytest.mark.parametrize("box", [
    None, {}, [], [0, 0, 1000], [0, 0, 1000, 1000, 1000],
    [True, 0, 1000, 1000], [0.0, 0, 1000, 1000], ["0", 0, 1000, 1000],
    [-1, 0, 1000, 1000], [0, 0, 1001, 1000], [0, 0, 10**500, 1000],
    [float("nan"), 0, 1000, 1000], [0, float("inf"), 1000, 1000],
    [100, 0, 100, 1000], [0, 100, 1000, 100], [900, 0, 100, 1000], [0, 900, 1000, 100],
])
def test_boxes_require_finite_ordered_integers_in_0_through_1000(box):
    payload = document()
    payload["objects"][0]["box_2d"] = box
    _, factory = fake_client(text=json.dumps(payload))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()


def test_excessive_object_count_is_rejected_instead_of_silently_truncated():
    _, factory = fake_client(text=json.dumps({"objects": document()["objects"] * 3}))
    result = GeminiObjectDetector(
        CloudSettings(), DetectionSettings(max_detections=2), client_factory=factory,
    ).detect(packet())
    assert result.status == "unavailable" and result.detections == ()


@pytest.mark.parametrize("maximum", [0, 21, True, 2.5])
def test_invalid_requested_limit_never_creates_a_client(maximum):
    _, factory = fake_client()
    result = GeminiObjectDetector(
        CloudSettings(), SimpleNamespace(max_detections=maximum), client_factory=factory,
    ).detect(packet())
    assert result.status == "unavailable" and "1 through 20" in result.error
    factory.assert_not_called()


def test_key_must_be_in_gemini_environment_variable_not_env_file_or_google_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "another-fake-key")
    (tmp_path / ".env").write_text(f"GEMINI_API_KEY={TEST_KEY}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _, factory = fake_client()
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and "Set GEMINI_API_KEY" in result.error
    factory.assert_not_called()


@pytest.mark.parametrize("image", [None, [], np.zeros((0, 2, 3), np.uint8),
    np.zeros((2, 2), np.uint8), np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32)])
def test_invalid_images_never_create_a_client(image):
    _, factory = fake_client()
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(
        FramePacket(8, 0, image)
    )
    assert result.status == "unavailable" and result.frame_id == 8
    factory.assert_not_called()


def test_missing_sdk_is_actionable_and_sanitized(monkeypatch):
    original_import = builtins.__import__

    def no_google(name, *args, **kwargs):
        if name == "google" or name.startswith("google.genai"):
            raise ImportError(f"PRIVATE SDK detail {TEST_KEY}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_google)
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings()).detect(packet())
    assert result.status == "unavailable" and "google-genai" in result.error
    assert TEST_KEY not in result.error and "PRIVATE" not in result.error


@pytest.mark.parametrize("status", ["incomplete", "requires_continuation", "requires_action", "failed", "cancelled"])
def test_unfinished_or_blocked_responses_never_return_partial_boxes(status):
    client, factory = fake_client(response=SimpleNamespace(status=status, output_text=json.dumps(document())))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()
    client.interactions.create.assert_called_once()
    client.close.assert_called_once_with()


def test_mapping_response_missing_status_is_unavailable_and_client_is_closed():
    client, factory = fake_client(response={"output_text": json.dumps(document())})
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()
    assert result.error == "Gemini returned no complete object detection result. Try another snapshot."
    client.close.assert_called_once_with()


def test_object_response_with_null_status_is_unavailable_and_client_is_closed():
    client, factory = fake_client(response=SimpleNamespace(status=None, output_text=json.dumps(document())))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()
    assert result.error == "Gemini returned no complete object detection result. Try another snapshot."
    client.close.assert_called_once_with()


def test_response_steps_are_supported_and_empty_text_is_unavailable():
    client, factory = fake_client(response={
        "status": "completed", "steps": [{"type": "model_output", "content": [
            {"type": "text", "text": json.dumps(document())},
        ]}],
    })
    detector = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory)
    assert detector.detect(packet()).status == "ready"
    client.interactions.create.return_value = SimpleNamespace(status="completed", output_text="")
    result = detector.detect(packet(8))
    assert result.status == "unavailable" and result.frame_id == 8 and result.detections == ()


@pytest.mark.parametrize("code,phrase", [(401, "access was denied"), (403, "access was denied"),
    (404, "model is unavailable"), (429, "quota"), (504, "timed out"), (503, "request failed")])
def test_service_failures_are_sanitized_and_not_retried(code, phrase, caplog):
    error = RuntimeError(f"PRIVATE service detail {TEST_KEY}")
    error.status_code = code
    client, factory = fake_client(error=error)
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and phrase in result.error
    assert TEST_KEY not in result.error and "PRIVATE" not in result.error
    assert caplog.text == ""
    client.interactions.create.assert_called_once()
    client.close.assert_called_once_with()


@pytest.mark.parametrize("error", [TimeoutError("PRIVATE timeout"), ConnectionError("PRIVATE network"),
    ImportError("PRIVATE dependency"), scene.SceneDescriptionError(f"PRIVATE helper {TEST_KEY}")])
def test_network_dependency_and_helper_errors_do_not_expose_raw_messages(error):
    client, factory = fake_client(error=error)
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and result.detections == ()
    assert TEST_KEY not in result.error and "PRIVATE" not in result.error
    client.close.assert_called_once_with()


def test_factory_failure_does_not_expose_raw_diagnostics():
    factory = Mock(side_effect=RuntimeError(f"PRIVATE factory {TEST_KEY}"))
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "unavailable" and "PRIVATE" not in result.error and TEST_KEY not in result.error


def test_ctrl_c_propagates_after_client_cleanup():
    client, factory = fake_client(error=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    client.close.assert_called_once_with()


def test_cleanup_error_is_ignored_without_logging_raw_data(caplog):
    client, factory = fake_client()
    client.close.side_effect = RuntimeError(f"PRIVATE cleanup {TEST_KEY}")
    result = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory).detect(packet())
    assert result.status == "ready" and caplog.text == ""


def test_each_frame_gets_a_fresh_stateless_request_and_never_stale_detections():
    client, factory = fake_client()
    client.interactions.create.side_effect = [
        SimpleNamespace(status="completed", output_text=json.dumps(document())),
        SimpleNamespace(status="completed", output_text='{"objects": []}'),
        TimeoutError("timeout"),
    ]
    detector = GeminiObjectDetector(CloudSettings(), DetectionSettings(), client_factory=factory)
    found, empty, failed = (detector.detect(packet(index)) for index in (10, 11, 12))
    assert found.frame_id == 10 and len(found.detections) == 1
    assert empty.frame_id == 11 and empty.status == "ready" and empty.detections == ()
    assert failed.frame_id == 12 and failed.status == "unavailable" and failed.detections == ()
    assert factory.call_count == 3 and client.close.call_count == 3
    for request in client.interactions.create.call_args_list:
        assert "previous_interaction_id" not in request.kwargs and request.kwargs["store"] is False


@pytest.mark.parametrize("status_code", [200, 429, 503])
def test_installed_sdk_object_json_contract_uses_only_local_mock_transport(monkeypatch, status_code, caplog):
    pytest.importorskip("google.genai", reason="Optional Gemini SDK is not installed")
    httpx = pytest.importorskip("httpx", reason="Optional SDK compatibility requires httpx")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    caplog.set_level("DEBUG")
    requests = []
    cloud = CloudSettings(timeout_s=3.0)
    payload = {
        "id": "offline-detection-interaction", "object": "interaction", "model": cloud.model,
        "status": "completed", "created": "2026-11-14T12:00:00Z", "updated": "2026-11-14T12:00:00Z",
        "steps": [{"type": "model_output", "content": [
            {"type": "text", "text": json.dumps(document())},
        ]}],
    }

    def respond(request):
        requests.append(request)
        if status_code == 200:
            return httpx.Response(200, json=payload)
        return httpx.Response(status_code, json={"error": {
            "code": "resource_exhausted" if status_code == 429 else "unavailable",
            "message": f"PRIVATE offline service detail {TEST_KEY}",
        }})

    def factory(*, api_key, http_options):
        assert api_key == TEST_KEY
        assert http_options == {"timeout": 3000, "retry_options": {"attempts": 0}}
        return scene._default_client_factory(
            api_key=api_key,
            http_options={
                **http_options, "base_url": "https://gemini-offline.invalid",
                "client_args": {"transport": httpx.MockTransport(respond), "trust_env": False},
            },
        )

    result = GeminiObjectDetector(cloud, DetectionSettings(max_detections=3), client_factory=factory).detect(packet())
    assert result.status == ("ready" if status_code == 200 else "unavailable")
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST" and request.url.host == "gemini-offline.invalid"
    assert request.url.path.endswith("/interactions")
    assert request.headers["x-goog-api-key"] == TEST_KEY
    assert request.extensions["timeout"]["read"] == 3.0
    body = json.loads(request.content)
    for name in ("store", "background", "stream"):
        assert body[name] is False
    assert body["tools"] == []
    assert body["generation_config"] == {"max_output_tokens": 1024, "tool_choice": "none"}
    assert body["response_format"] == {
        "type": "text", "mime_type": "application/json", "schema": gemini._response_schema(3),
    }
    assert "previous_interaction_id" not in body and "automatic_continuation" not in body
    content = body["input"][0]["content"]
    assert body["input"][0]["type"] == "user_input"
    image = content[1]
    assert image["type"] == "image" and image["mime_type"] == "image/jpeg" and "uri" not in image
    data = base64.b64decode(image["data"], validate=True)
    assert data.startswith(b"\xff\xd8") and len(data) <= scene.MAX_IMAGE_BYTES
    assert TEST_KEY not in caplog.text and "PRIVATE" not in caplog.text and image["data"] not in caplog.text
