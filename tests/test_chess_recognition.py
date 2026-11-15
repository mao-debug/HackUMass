"""Chess observations use strict synthetic responses and no real camera/API."""

from dataclasses import asdict
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vidi.ai import chess_recognition as vision
from vidi.core.chess_types import SQUARE_NAMES
from vidi.core.configuration import CloudSettings
from vidi.core.types import FramePacket

KEY = "synthetic-chess-key-not-a-credential"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    def forbidden(*args, **kwargs):
        pytest.fail("A chess observation test attempted a real network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def settings(**kwargs):
    return SimpleNamespace(backend="gemini", fixture_path=None, orientation="white_near", **kwargs)


def packet():
    return FramePacket(10, time.monotonic_ns(), np.zeros((120, 160, 3), np.uint8))


def payload():
    return {"board_visible": True, "orientation_confirmed": True,
            "squares": {name: "." for name in SQUARE_NAMES}}


def client_for(text=None, response=None, error=None):
    response = response if response is not None else {"status": "completed", "output_text": text or json.dumps(payload())}
    client = SimpleNamespace(interactions=SimpleNamespace(create=Mock(return_value=response, side_effect=error)), close=Mock())
    return client, Mock(return_value=client)


def test_constructor_never_creates_client_or_reads_key(monkeypatch):
    factory = Mock(side_effect=AssertionError("Client created eagerly"))
    monkeypatch.setattr(vision, "os", SimpleNamespace(environ=SimpleNamespace(get=lambda *_args: pytest.fail("Key read"))))
    vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory)
    factory.assert_not_called()


def test_complete_response_is_an_observation_not_a_move_and_has_bounded_request():
    client, factory = client_for()
    frame = packet()
    result = vision.GeminiChessObserver(CloudSettings(timeout_s=3), settings(), client_factory=factory).observe(frame)
    assert result.status == "ready" and result.squares == (".",) * 64
    assert result.frame_id == frame.frame_id and result.captured_monotonic_ns == frame.captured_monotonic_ns
    assert not result.simulated and "move" not in asdict(result)
    request = client.interactions.create.call_args.kwargs
    assert request["tools"] == [] and request["store"] is False and request["automatic_continuation"] is False
    assert request["generation_config"] == {"max_output_tokens": 2048, "tool_choice": "none"}
    assert request["input"][1]["mime_type"] == "image/jpeg" and "uri" not in request["input"][1]
    assert "a1 bottom-left" in request["input"][0]["text"]
    factory.assert_called_once_with(api_key=KEY, http_options={"timeout": 3000, "retry_options": {"attempts": 0}})
    client.close.assert_called_once_with()


@pytest.mark.parametrize("change", ["hidden", "orientation", "unknown"])
def test_partial_or_uncertain_views_cannot_be_ready(change):
    document = payload()
    if change == "hidden":
        document["board_visible"] = False
    elif change == "orientation":
        document["orientation_confirmed"] = False
    else:
        document["squares"]["e4"] = "?"
    client, factory = client_for(text=json.dumps(document))
    result = vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory).observe(packet())
    assert result.status == "unclear" and result.error and len(result.squares) == 64


@pytest.mark.parametrize("change", ["missing_square", "extra_square", "extra_field", "bad_piece", "bool_string", "piece_object"])
def test_invalid_payloads_never_produce_a_board(change):
    document = payload()
    if change == "missing_square":
        del document["squares"]["a1"]
    elif change == "extra_square":
        document["squares"]["i9"] = "Q"
    elif change == "extra_field":
        document["best_move"] = "e2e4"
    elif change == "bad_piece":
        document["squares"]["a1"] = "white rook"
    elif change == "bool_string":
        document["board_visible"] = "true"
    else:
        document["squares"]["a1"] = {"piece": "R"}
    _, factory = client_for(text=json.dumps(document))
    result = vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory).observe(packet())
    assert result.status == "failed" and not result.squares


@pytest.mark.parametrize("text", [
    '{"board_visible":true,"board_visible":false}',
    json.dumps(payload()).replace('"a1": "."', '"a1": ".", "a1": "R"'),
    json.dumps(payload()).replace('"a1": "."', '"a1": NaN'),
    "x" * (vision.MAX_REPLY_CHARACTERS + 1), KEY,
])
def test_duplicate_nonfinite_oversize_and_secret_payloads_are_rejected(text, caplog):
    _, factory = client_for(text=text)
    result = vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory).observe(packet())
    assert result.status == "failed" and KEY not in result.error + caplog.text


@pytest.mark.parametrize("status", [None, "incomplete", "in_progress", "failed"])
def test_only_explicit_completed_responses_can_be_observations(status):
    _, factory = client_for(response={"status": status, "output_text": json.dumps(payload())})
    assert vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory).observe(packet()).status == "failed"


@pytest.mark.parametrize("stage", ["before", "encoding", "client", "response"])
def test_cancellation_prevents_late_upload_or_observation(monkeypatch, stage):
    event = threading.Event()
    client, factory = client_for()
    def encode(*args):
        if stage == "encoding":
            event.set()
        return "synthetic-image"
    def make_client(**kwargs):
        if stage == "client":
            event.set()
        return client
    def request(**kwargs):
        if stage == "response":
            event.set()
        return {"status": "completed", "output_text": json.dumps(payload())}
    monkeypatch.setattr(vision, "_encode_image", encode)
    factory.side_effect = make_client
    client.interactions.create.side_effect = request
    if stage == "before":
        event.set()
    result = vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory).observe(packet(), cancel_event=event)
    assert result.status == "cancelled" and not result.squares
    assert client.interactions.create.call_count == int(stage == "response")
    assert client.close.call_count == int(stage in ("client", "response"))


def test_missing_key_and_service_failures_are_sanitized(monkeypatch, caplog):
    client, factory = client_for(error=TimeoutError("private " + KEY))
    observer = vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory)
    result = observer.observe(packet())
    assert result.status == "failed" and "timed out" in result.error
    assert "private" not in result.error and KEY not in result.error + caplog.text
    assert client.interactions.create.call_count == 1
    client.close.assert_called_once()
    factory.reset_mock()
    monkeypatch.delenv("GEMINI_API_KEY")
    assert observer.observe(packet()).status == "failed"
    factory.assert_not_called()


def test_fixture_is_explicit_bounded_ordered_simulation_and_repeats_last(monkeypatch):
    fixture = Path(__file__).resolve().parents[1] / "examples/chess-fixture.json"
    configured = SimpleNamespace(backend="fixture", fixture_path=fixture, orientation="white_near")
    monkeypatch.setattr(vision, "os", SimpleNamespace(environ=SimpleNamespace(get=lambda *_args: pytest.fail("Key read in fixture"))))
    observer = vision.create_chess_observer(CloudSettings(), configured)
    results = [observer.observe(packet()) for _ in range(8)]
    assert all(item.status == "ready" and item.simulated for item in results)
    e2, e4 = SQUARE_NAMES.index("e2"), SQUARE_NAMES.index("e4")
    assert results[0].squares[e2] == "P" and results[1].squares[e4] == "P"
    assert results[-1].squares == results[-2].squares


def test_bad_fixture_orientation_and_shape_fail_without_network(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"v": 1, "orientation": "black_near", "observations": [payload()]}))
    with pytest.raises(vision.ChessObservationError):
        vision.FixtureChessObserver(path, settings())
    path.write_text('{"v":1,"v":2}')
    with pytest.raises(vision.ChessObservationError):
        vision.FixtureChessObserver(path, settings())


def test_installed_sdk_serializes_chess_schema_offline(monkeypatch):
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "offline-chess", "object": "interaction", "status": "completed",
            "model": "test-chess", "created": "2026-11-15T12:00:00Z", "updated": "2026-11-15T12:00:00Z",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": json.dumps(payload())}]}]})
    def factory(*, api_key, http_options):
        return vision._default_client_factory(api_key=api_key, http_options={**http_options,
            "base_url": "https://offline-chess.invalid", "client_args": {"transport": httpx.MockTransport(respond), "trust_env": False}})
    result = vision.GeminiChessObserver(CloudSettings(), settings(), client_factory=factory).observe(packet())
    assert result.status == "ready" and len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["response_format"]["schema"]["properties"]["squares"]["required"] == list(SQUARE_NAMES)
    assert body["store"] is False and body["tools"] == [] and "previous_interaction_id" not in body
