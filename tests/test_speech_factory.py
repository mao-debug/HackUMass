"""Speech selection is explicit, cancellable, and inert until called."""

from unittest.mock import Mock

from vidi.audio import speech_factory
from vidi.audio import text_to_speech, piper_speech, gemini_speech
from vidi.core.configuration import SpeechSettings
from vidi.core.types import SpeechResult


def test_factory_does_not_initialize_any_engine_until_speech(monkeypatch):
    local = Mock()
    warm = Mock()
    cloud = Mock()
    monkeypatch.setattr(text_to_speech, "LocalSpeech", local)
    monkeypatch.setattr(piper_speech, "WarmPiperSpeech", warm)
    monkeypatch.setattr(gemini_speech, "GeminiSpeech", cloud)
    for backend in (None, "espeak_ng", "piper", "gemini"):
        session = speech_factory.create_speech(SpeechSettings(), backend)
        session.close()
    local.assert_not_called()
    warm.assert_not_called()
    cloud.assert_not_called()


def test_selected_engine_is_persistent_and_cloud_is_never_an_automatic_fallback(monkeypatch):
    warm_engine = Mock()
    warm_engine.speak.return_value = SpeechResult(status="failed", backend="piper")
    warm = Mock(return_value=warm_engine)
    cloud = Mock()
    monkeypatch.setattr(piper_speech, "WarmPiperSpeech", warm)
    monkeypatch.setattr(gemini_speech, "GeminiSpeech", cloud)
    session = speech_factory.create_speech(SpeechSettings(), "piper")
    assert session.speak("First.").status == "failed"
    assert session.speak("Second.").status == "failed"
    warm.assert_called_once()
    cloud.assert_not_called()
    session.stop()
    warm_engine.stop.assert_called_once()
    session.close()
    warm_engine.close.assert_called_once()


def test_cloud_enablement_is_explicit_and_cancel_is_forwarded(monkeypatch):
    cloud_engine = Mock()
    cloud = Mock(return_value=cloud_engine)
    monkeypatch.setattr(gemini_speech, "GeminiSpeech", cloud)
    session = speech_factory.create_speech(SpeechSettings(), "gemini", cloud_enabled=True)

    def while_speaking(text, *, cancel_event, output_path):
        assert not cancel_event.is_set()
        session.stop()
        assert cancel_event.is_set()
        return SpeechResult(status="cancelled", backend="gemini")

    cloud_engine.speak.side_effect = while_speaking
    result = session.speak("A selected cloud utterance.")
    assert result.status == "cancelled"
    assert cloud.call_args.kwargs == {"cloud_enabled": True}
    cloud_engine.stop.assert_called_once()


def test_coordinate_speech_changes_no_move_notation_or_embedded_identifiers():
    raw = "White knight from g1 to f3. e4 is empty; xg1suffix is an identifier."
    rendered = speech_factory.render_chess_speech(raw)
    assert rendered == "White knight from G one to F three. E four is empty; xg1suffix is an identifier."
    assert raw == "White knight from g1 to f3. e4 is empty; xg1suffix is an identifier."
