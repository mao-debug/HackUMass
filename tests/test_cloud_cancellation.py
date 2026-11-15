"""Cancellation prevents delayed image preparation from submitting an upload."""

from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest

from vidi.ai import gemini_text_recognition as reading
from vidi.ai import scene_description as scene
from vidi.core.configuration import CloudSettings, OcrSettings
from vidi.core.types import FramePacket


@pytest.mark.parametrize("kind", ["scene", "reading"])
@pytest.mark.parametrize("stage", ["before", "encoding", "client", "response"])
def test_image_upload_cancellation_at_each_boundary(monkeypatch, kind, stage):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-offline-cancellation-key")
    cancel = threading.Event()
    module = scene if kind == "scene" else reading
    image = np.zeros((20, 30, 3), np.uint8)
    response = {"status": "completed", "output_text": "A chair." if kind == "scene" else '{"text":"A sign"}'}

    def encode(*args):
        if stage == "encoding":
            cancel.set()
        return "synthetic-image-data"

    def request(**kwargs):
        if stage == "response":
            cancel.set()
        return response

    client = SimpleNamespace(interactions=SimpleNamespace(create=Mock(side_effect=request)), close=Mock())

    def factory(**kwargs):
        if stage == "client":
            cancel.set()
        return client

    factory = Mock(side_effect=factory)
    monkeypatch.setattr(module, "_encode_image", encode)
    if stage == "before":
        cancel.set()

    if kind == "scene":
        adapter = scene.GeminiSceneDescriber(CloudSettings(), client_factory=factory)
        with pytest.raises(scene.SceneDescriptionError, match="cancelled"):
            adapter.describe(image, cancel_event=cancel)
    else:
        adapter = reading.GeminiTextRecognizer(CloudSettings(), OcrSettings(), client_factory=factory)
        result = adapter.recognize(FramePacket(1, 100, image), cancel_event=cancel)
        assert result.status == "cancelled" and result.text == ""

    assert client.interactions.create.call_count == int(stage == "response")
    assert factory.call_count == int(stage in ("client", "response"))
    assert client.close.call_count == factory.call_count
