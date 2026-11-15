"""Two rectified images share one mocked request; disagreements never become ready."""

import json
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vidi.ai import chess_recognition as vision
from vidi.core.chess_types import SQUARE_NAMES
from vidi.core.configuration import ChessSettings, CloudSettings
from vidi.core.types import FramePacket

KEY = "synthetic-pair-key-not-a-credential"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: pytest.fail("Real network attempted"))
    monkeypatch.setattr(socket.socket, "connect", lambda *_args, **_kwargs: pytest.fail("Real network attempted"))


def frames():
    now = time.monotonic_ns()
    return (FramePacket(1, now - 500_000_000, np.zeros((400, 400, 3), np.uint8)),
            FramePacket(2, now - 100_000_000, np.full((400, 400, 3), 255, np.uint8)))


def document():
    return {name: {"board_visible": True, "orientation_confirmed": True,
                   "squares": {square: "." for square in SQUARE_NAMES}} for name in ("first", "second")}


def observer(document_override=None, response=None):
    response = response or {"status": "completed", "output_text": json.dumps(document_override or document())}
    client = SimpleNamespace(interactions=SimpleNamespace(create=Mock(return_value=response)), close=Mock())
    factory = Mock(return_value=client)
    return vision.GeminiChessObserver(CloudSettings(timeout_s=3), ChessSettings(), client_factory=factory), client, factory


def test_one_request_contains_two_distinct_images_with_no_retries_or_tools():
    subject, client, factory = observer()
    first, second = frames()
    results = subject.observe_pair(first, second)
    assert [item.status for item in results] == ["ready", "ready"]
    assert [item.frame_id for item in results] == [1, 2]
    assert [item.captured_monotonic_ns for item in results] == [first.captured_monotonic_ns, second.captured_monotonic_ns]
    request = client.interactions.create.call_args.kwargs
    assert len(request["input"]) == 3 and request["input"][1]["data"] != request["input"][2]["data"]
    assert request["tools"] == [] and request["store"] is False and request["automatic_continuation"] is False
    assert request["response_format"]["schema"]["required"] == ["first", "second"]
    factory.assert_called_once_with(api_key=KEY, http_options={"timeout": 3000, "retry_options": {"attempts": 0}})
    client.interactions.create.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize("change", ["disagree", "unknown", "hidden", "orientation", "missing", "extra", "duplicate", "secret", "oversize"])
def test_pair_disagreement_and_invalid_responses_never_ready(change):
    payload = document()
    if change == "disagree":
        payload["second"]["squares"]["e4"] = "P"
    elif change == "unknown":
        payload["second"]["squares"]["e4"] = "?"
    elif change == "hidden":
        payload["first"]["board_visible"] = False
    elif change == "orientation":
        payload["second"]["orientation_confirmed"] = False
    elif change == "missing":
        payload["first"]["squares"].pop("h8")
    elif change == "extra":
        payload["move"] = "e2e4"
    text = json.dumps(payload)
    if change == "duplicate":
        text = text.replace('"a1": "."', '"a1": ".", "a1": "R"')
    elif change == "secret":
        text = KEY
    elif change == "oversize":
        text = "x" * (vision.MAX_REPLY_CHARACTERS + 1)
    subject, client, _ = observer(response={"status": "completed", "output_text": text})
    results = subject.observe_pair(*frames())
    assert all(item.status in ("unclear", "failed") and KEY not in item.error for item in results)
    client.close.assert_called_once()


@pytest.mark.parametrize("change", ["same_frame", "too_close", "stale", "future"])
def test_invalid_frame_pair_abstains_before_client_or_key(monkeypatch, change):
    first, second = frames()
    if change == "same_frame":
        second = first
    elif change == "too_close":
        second = FramePacket(2, first.captured_monotonic_ns + 299_999_999, second.image_bgr)
    elif change == "stale":
        first = FramePacket(1, time.monotonic_ns() - 31_000_000_000, first.image_bgr)
    else:
        second = FramePacket(2, time.monotonic_ns() + 10_000_000_000, second.image_bgr)
    subject, _, factory = observer()
    monkeypatch.setattr(vision, "os", SimpleNamespace(environ=SimpleNamespace(get=lambda *_: pytest.fail("Key read"))))
    assert all(item.status == "unclear" for item in subject.observe_pair(first, second))
    factory.assert_not_called()


@pytest.mark.parametrize("stage", ["before", "encoding", "client", "response"])
def test_cancel_never_returns_late_pair_or_leaks_client(monkeypatch, stage):
    subject, client, factory = observer()
    event = threading.Event()
    def encode(*args):
        if stage == "encoding":
            event.set()
        return "synthetic"
    def make_client(**kwargs):
        if stage == "client":
            event.set()
        return client
    def request(**kwargs):
        if stage == "response":
            event.set()
        return {"status": "completed", "output_text": json.dumps(document())}
    factory.side_effect = make_client
    client.interactions.create.side_effect = request
    monkeypatch.setattr(vision, "_encode_image", encode)
    if stage == "before":
        event.set()
    assert all(item.status == "cancelled" for item in subject.observe_pair(*frames(), cancel_event=event))
    assert client.interactions.create.call_count == int(stage == "response")
    assert client.close.call_count == int(stage in ("client", "response"))


def test_expiry_during_cloud_call_abstains_after_closing_client(monkeypatch):
    subject, client, _ = observer()
    first, second = frames()
    def request(**kwargs):
        monkeypatch.setattr(vision.time, "monotonic_ns", lambda: first.captured_monotonic_ns + 31_000_000_000)
        return {"status": "completed", "output_text": json.dumps(document())}
    client.interactions.create.side_effect = request
    assert all(item.status == "unclear" for item in subject.observe_pair(first, second))
    client.close.assert_called_once()


def test_service_timeout_and_missing_key_are_sanitized(monkeypatch):
    subject, client, factory = observer()
    client.interactions.create.side_effect = TimeoutError(KEY)
    results = subject.observe_pair(*frames())
    assert all(item.status == "failed" and KEY not in item.error and "timed out" in item.error for item in results)
    client.close.assert_called_once()
    factory.reset_mock()
    monkeypatch.delenv("GEMINI_API_KEY")
    assert all(item.status == "failed" for item in subject.observe_pair(*frames()))
    factory.assert_not_called()


def test_installed_sdk_serializes_two_images_and_nested_schema_offline(monkeypatch):
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "offline-pair", "object": "interaction", "status": "completed",
            "model": "test-pair", "created": "2026-11-15T12:00:00Z", "updated": "2026-11-15T12:00:00Z",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": json.dumps(document())}]}]})
    def factory(*, api_key, http_options):
        return vision._default_client_factory(api_key=api_key, http_options={**http_options,
            "base_url": "https://offline-pair.invalid", "client_args": {"transport": httpx.MockTransport(respond), "trust_env": False}})
    subject = vision.GeminiChessObserver(CloudSettings(), ChessSettings(), client_factory=factory)
    assert all(item.status == "ready" for item in subject.observe_pair(*frames()))
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["input"][0]["type"] == "user_input"
    assert sum(item["type"] == "image" for item in body["input"][0]["content"]) == 2
    assert body["response_format"]["schema"]["properties"]["second"]["properties"]["squares"]["required"] == list(SQUARE_NAMES)
    assert body["store"] is False and body["tools"] == [] and "previous_interaction_id" not in body
