"""Gemini boundary tests using fake clients only; never contact a service."""

import base64
import builtins
from dataclasses import FrozenInstanceError
import json
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.ai import scene_description as scene
from vidi.ai.scene_description import (
    GeminiSceneDescriber,
    SceneDescriptionError,
    SceneDescriptionResult,
)
from vidi.core.configuration import CloudSettings


TEST_KEY = "test-gemini-key-not-a-real-credential"


@pytest.fixture(autouse=True)
def isolated_credential(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", TEST_KEY)


@pytest.fixture
def image():
    return np.full((120, 200, 3), (20, 80, 180), dtype=np.uint8)


def fake_client(response=None, error=None):
    if response is None:
        response = SimpleNamespace(status="completed", output_text="A chair appears beside a table.")
    create = Mock(return_value=response, side_effect=error)
    client = SimpleNamespace(interactions=SimpleNamespace(create=create), close=Mock())
    factory = Mock(return_value=client)
    return client, factory


def test_constructor_does_not_read_credentials_or_create_client(monkeypatch):
    environment = SimpleNamespace(get=Mock(side_effect=AssertionError("Credentials read too early")))
    monkeypatch.setattr(scene, "os", SimpleNamespace(environ=environment))
    factory = Mock(side_effect=AssertionError("Client created too early"))

    GeminiSceneDescriber(CloudSettings(), client_factory=factory)

    environment.get.assert_not_called()
    factory.assert_not_called()


def test_default_sdk_client_is_created_only_on_describe(monkeypatch, image):
    client, _ = fake_client()
    constructor = Mock(return_value=client)
    google = ModuleType("google")

    def options_factory(**options):
        return SimpleNamespace(
            timeout=options["timeout"],
            retry_options=SimpleNamespace(attempts=options["retry_options"]["attempts"]),
        )

    google.genai = SimpleNamespace(
        Client=constructor, types=SimpleNamespace(HttpOptions=options_factory)
    )
    monkeypatch.setitem(sys.modules, "google", google)
    describer = GeminiSceneDescriber(CloudSettings())
    constructor.assert_not_called()

    describer.describe(image)

    constructor.assert_called_once()
    options = constructor.call_args.kwargs
    assert options["api_key"] == TEST_KEY
    assert options["vertexai"] is False
    assert options["http_options"].timeout == 15000
    assert options["http_options"].retry_options.attempts == 0
    client.close.assert_called_once_with()


def test_missing_optional_sdk_has_sanitized_actionable_error(monkeypatch, image):
    original_import = builtins.__import__

    def no_google(name, *args, **kwargs):
        if name == "google" or name.startswith("google.genai"):
            raise ImportError(f"Private SDK diagnostic {TEST_KEY}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_google)

    with pytest.raises(SceneDescriptionError, match="optional google-genai") as caught:
        GeminiSceneDescriber(CloudSettings()).describe(image)

    assert TEST_KEY not in str(caught.value)
    assert "Private SDK diagnostic" not in str(caught.value)


def test_only_gemini_environment_key_is_accepted(monkeypatch, image):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "another-provider-key")
    client, factory = fake_client()

    with pytest.raises(SceneDescriptionError, match="Set GEMINI_API_KEY"):
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    factory.assert_not_called()
    client.interactions.create.assert_not_called()


def test_request_is_stateless_bounded_and_disables_tools_and_retries(monkeypatch, image):
    client, factory = fake_client()
    settings = CloudSettings(timeout_s=4.25, max_output_tokens=128)
    monkeypatch.setattr(scene, "time", SimpleNamespace(monotonic=Mock(side_effect=[10.0, 11.25])))

    result = GeminiSceneDescriber(settings, client_factory=factory).describe(image, "  What is here?  ")

    factory.assert_called_once_with(
        api_key=TEST_KEY,
        http_options={"timeout": 4250, "retry_options": {"attempts": 0}},
    )
    request = client.interactions.create.call_args.kwargs
    assert request["model"] == settings.model
    assert request["input"][0] == {"type": "text", "text": "What is here?"}
    assert request["input"][1]["type"] == "image"
    assert request["input"][1]["mime_type"] == "image/jpeg"
    assert request["store"] is False
    assert request["background"] is False
    assert request["stream"] is False
    assert request["automatic_continuation"] is False
    assert request["timeout"] == 4.25
    assert request["generation_config"] == {"max_output_tokens": 128, "tool_choice": "none"}
    assert request["response_format"] == {"type": "text", "mime_type": "text/plain"}
    assert request["tools"] == []
    assert "previous_interaction_id" not in request
    assert "uri" not in request["input"][1]
    for instruction in (
        "two short sentences",
        "State uncertainty",
        "precise distances",
        "navigation instructions",
        "sensitive personal attributes",
        "untrusted observation",
        "Do not use tools",
    ):
        assert instruction in request["system_instruction"]
    assert result == SceneDescriptionResult("A chair appears beside a table.", settings.model, 1.25)
    client.close.assert_called_once_with()
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


def test_image_upload_is_resized_jpeg_with_correct_colors_and_no_source_mutation():
    image = np.full((1200, 2000, 3), (20, 80, 180), dtype=np.uint8)
    original = image.copy()
    client, factory = fake_client()

    GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    data = base64.b64decode(client.interactions.create.call_args.kwargs["input"][1]["data"], validate=True)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert data.startswith(b"\xff\xd8")
    assert len(data) <= 2 * 1024 * 1024
    assert decoded.shape[:2] == (614, 1024)
    np.testing.assert_allclose(decoded[100, 100], image[100, 100], atol=4)
    np.testing.assert_array_equal(image, original)


def test_configured_image_edge_is_honored_and_small_images_are_not_upscaled(image):
    client, factory = fake_client()
    GeminiSceneDescriber(CloudSettings(max_image_edge=100), client_factory=factory).describe(image)
    data = base64.b64decode(client.interactions.create.call_args.kwargs["input"][1]["data"])
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (60, 100)

    GeminiSceneDescriber(CloudSettings(max_image_edge=1024), client_factory=factory).describe(image)
    data = base64.b64decode(client.interactions.create.call_args.kwargs["input"][1]["data"])
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == image.shape[:2]


@pytest.mark.parametrize(
    "invalid_image",
    [None, [], np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2), np.uint8),
     np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32)],
)
def test_invalid_image_is_rejected_before_creating_client(invalid_image):
    _, factory = fake_client()

    with pytest.raises(SceneDescriptionError, match="camera image"):
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(invalid_image)

    factory.assert_not_called()


@pytest.mark.parametrize("question", ["", "   ", None, 123, "x" * 501])
def test_invalid_question_is_rejected_before_creating_client(image, question):
    _, factory = fake_client()

    with pytest.raises(SceneDescriptionError, match="scene question"):
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image, question)

    factory.assert_not_called()


@pytest.mark.parametrize("failure", ["encoder-false", "encoder-error", "oversized"])
def test_failed_or_oversized_encoding_never_creates_service_client(monkeypatch, image, failure):
    if failure == "encoder-error":
        encoder = Mock(side_effect=cv2.error(f"Private image diagnostic {TEST_KEY}"))
    elif failure == "oversized":
        encoder = Mock(return_value=(True, np.zeros(scene.MAX_IMAGE_BYTES + 1, np.uint8)))
    else:
        encoder = Mock(return_value=(False, None))
    monkeypatch.setattr(cv2, "imencode", encoder)
    _, factory = fake_client()

    with pytest.raises(SceneDescriptionError) as caught:
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert TEST_KEY not in str(caught.value)
    assert "Private image diagnostic" not in str(caught.value)
    factory.assert_not_called()
    assert encoder.call_count == (3 if failure == "oversized" else 1)


def test_current_step_response_reads_model_text_only(image):
    response = {
        "status": "completed",
        "steps": [
            {"type": "thought", "content": [{"type": "text", "text": "Private thought"}]},
            {"type": "function_call", "text": "Do not execute this"},
            {"type": "model_output", "content": [
                {"type": "image", "data": "not text"},
                {"type": "text", "text": "A person appears near a chair."},
            ]},
        ],
    }
    client, factory = fake_client(response)

    result = GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert result.text == "A person appears near a chair."
    client.interactions.create.assert_called_once()


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete", "requires_action", "requires_continuation"])
def test_noncompleted_responses_do_not_return_partial_or_blocked_text(image, status):
    client, factory = fake_client(SimpleNamespace(status=status, output_text=f"Private result {TEST_KEY}"))

    with pytest.raises(SceneDescriptionError) as caught:
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert TEST_KEY not in str(caught.value)
    assert "Private result" not in str(caught.value)
    client.interactions.create.assert_called_once()
    client.close.assert_called_once_with()


@pytest.mark.parametrize("text", [None, "", " \n\t ", "\x00\x01"])
def test_empty_or_unusable_responses_have_controlled_error(image, text):
    client, factory = fake_client(SimpleNamespace(status="completed", output_text=text))

    with pytest.raises(SceneDescriptionError, match="no .*description"):
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    client.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("First sentence. Second sentence. Third sentence.", "First sentence. Second sentence."),
     (f"A sign shows {TEST_KEY}.", "A sign shows [redacted].")],
)
def test_response_is_brief_and_credentials_are_redacted(image, text, expected):
    _, factory = fake_client(SimpleNamespace(status="completed", output_text=text))

    result = GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert result.text == expected


@pytest.mark.parametrize("text", [" ".join(["word"] * 200), "x" * 2000])
def test_long_responses_have_hard_local_length_limits(image, text):
    _, factory = fake_client(SimpleNamespace(status="completed", output_text=text))

    result = GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert len(result.text.split()) <= 45
    assert len(result.text) <= 320
    assert result.text.endswith("…")


@pytest.mark.parametrize(
    ("code", "expected"),
    [(401, "access was denied"), (403, "access was denied"), (429, "quota"),
     (404, "model is unavailable"), (504, "timed out"), (500, "request failed")],
)
def test_service_errors_never_expose_credentials_or_raw_diagnostics(image, caplog, code, expected):
    error = RuntimeError(f"Private service diagnostic {TEST_KEY}")
    error.status_code = code
    client, factory = fake_client(error=error)

    with pytest.raises(SceneDescriptionError, match=expected) as caught:
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert TEST_KEY not in str(caught.value)
    assert "Private service diagnostic" not in str(caught.value)
    assert caplog.text == ""
    client.interactions.create.assert_called_once()
    client.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("error_class", "expected"), [(TimeoutError, "timed out"), (ConnectionError, "unreachable")]
)
def test_network_failures_are_actionable_without_automatic_retry(image, error_class, expected):
    client, factory = fake_client(error=error_class(f"Private network diagnostic {TEST_KEY}"))

    with pytest.raises(SceneDescriptionError, match=expected) as caught:
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert TEST_KEY not in str(caught.value)
    client.interactions.create.assert_called_once()
    client.close.assert_called_once_with()


def test_client_creation_failure_is_sanitized(image):
    factory = Mock(side_effect=RuntimeError(f"Private client diagnostic {TEST_KEY}"))

    with pytest.raises(SceneDescriptionError, match="request failed") as caught:
        GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert TEST_KEY not in str(caught.value)
    assert "Private client diagnostic" not in str(caught.value)
    factory.assert_called_once()


def test_cleanup_failure_cannot_expose_raw_error_or_hide_description(image, caplog):
    client, factory = fake_client()
    client.close.side_effect = RuntimeError(f"Private cleanup diagnostic {TEST_KEY}")

    result = GeminiSceneDescriber(CloudSettings(), client_factory=factory).describe(image)

    assert result.text == "A chair appears beside a table."
    assert caplog.text == ""


def test_each_request_uses_new_client_without_history(image):
    client, factory = fake_client()
    describer = GeminiSceneDescriber(CloudSettings(), client_factory=factory)

    describer.describe(image)
    describer.describe(image)

    assert factory.call_count == 2
    assert client.close.call_count == 2
    assert client.interactions.create.call_count == 2
    for call in client.interactions.create.call_args_list:
        assert "previous_interaction_id" not in call.kwargs
        assert call.kwargs["store"] is False


@pytest.mark.parametrize("status_code", [200, 429, 503])
def test_installed_sdk_serializes_and_parses_using_only_local_mock_transport(
    monkeypatch, image, status_code, caplog
):
    """Exercise the real optional SDK with every external socket path blocked."""
    def deny_network(*args, **kwargs):
        pytest.fail("SDK compatibility tests must never open an external connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    pytest.importorskip("google.genai", reason="Optional Gemini SDK is not installed")
    httpx = pytest.importorskip("httpx", reason="The SDK compatibility check needs httpx")
    caplog.set_level("DEBUG")
    requests = []
    settings = CloudSettings(timeout_s=3.0, max_output_tokens=128)
    payload = {
        "id": "offline-test-interaction",
        "object": "interaction",
        "model": settings.model,
        "status": "completed",
        "created": "2026-11-14T12:00:00Z",
        "updated": "2026-11-14T12:00:00Z",
        "steps": [{"type": "model_output", "content": [
            {"type": "text", "text": "A chair appears beside a table."},
        ]}],
    }

    def respond(request):
        requests.append(request)
        if status_code == 200:
            return httpx.Response(200, json=payload)
        return httpx.Response(
            status_code,
            json={"error": {
                "code": "resource_exhausted" if status_code == 429 else "unavailable",
                "message": f"Private offline service diagnostic {TEST_KEY}",
            }},
        )

    def factory(*, api_key, http_options):
        assert api_key == TEST_KEY
        assert http_options == {"timeout": 3000, "retry_options": {"attempts": 0}}
        return scene._default_client_factory(
            api_key=api_key,
            http_options={
                **http_options,
                "base_url": "https://gemini-offline.invalid",
                "client_args": {"transport": httpx.MockTransport(respond), "trust_env": False},
            },
        )

    describer = GeminiSceneDescriber(settings, client_factory=factory)
    if status_code == 200:
        result = describer.describe(image)
        assert result.text == "A chair appears beside a table."
        assert result.model == settings.model
    else:
        with pytest.raises(SceneDescriptionError) as caught:
            describer.describe(image)
        assert TEST_KEY not in str(caught.value)
        assert "Private offline service diagnostic" not in str(caught.value)

    # Transient failures must be attempted once, including in the real SDK's
    # retry layer; checking only our injectable-factory arguments misses this.
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.host == "gemini-offline.invalid"
    assert request.url.path.endswith("/interactions")
    assert request.headers["x-goog-api-key"] == TEST_KEY
    assert request.extensions["timeout"]["read"] == 3.0
    body = json.loads(request.content)
    assert body["model"] == settings.model
    assert body["store"] is False
    assert body["background"] is False
    assert body["stream"] is False
    assert body["tools"] == []
    assert body["generation_config"] == {"max_output_tokens": 128, "tool_choice": "none"}
    assert body["response_format"] == {"type": "text", "mime_type": "text/plain"}
    assert "previous_interaction_id" not in body
    assert "automatic_continuation" not in body  # A client-side option, not wire data.
    # The real SDK normalizes the content list into a user-input timeline step.
    assert len(body["input"]) == 1
    assert body["input"][0]["type"] == "user_input"
    content = body["input"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is in front of me?"}
    image_input = content[1]
    assert image_input["type"] == "image"
    assert image_input["mime_type"] == "image/jpeg"
    assert "uri" not in image_input
    data = base64.b64decode(image_input["data"], validate=True)
    assert data.startswith(b"\xff\xd8")
    assert len(data) <= scene.MAX_IMAGE_BYTES
    assert TEST_KEY not in caplog.text
    assert "Private offline service diagnostic" not in caplog.text
    assert image_input["data"] not in caplog.text
