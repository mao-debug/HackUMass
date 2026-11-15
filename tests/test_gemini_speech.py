"""Gemini speech tests use generated PCM and local clients; never audible output."""

import base64
import io
import json
import socket
import stat
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from vidi.ai import scene_description as scene
from vidi.audio import gemini_speech as speech
from vidi.core.configuration import SpeechSettings


KEY = "offline-tts-test-key-never-a-real-credential"
PCM = b"\x10\x00" * 480


def delta(data=PCM, **changes):
    return {"event_type": "step.delta", "index": 0, "delta": {
        "type": "audio", "mime_type": "audio/l16", "sample_rate": 24000, "channels": 1,
        "data": base64.b64encode(data).decode() if isinstance(data, bytes) else data, **changes,
    }}


def complete(status="completed"):
    return {"event_type": "interaction.completed", "interaction": {"id": "offline-tts", "status": status}}


class Stream:
    def __init__(self, events):
        self.events = events
        self.close = Mock()
    def __iter__(self):
        return iter(self.events)


def client_for(events=None, error=None):
    stream = Stream([delta(), complete()] if events is None else events)
    client = SimpleNamespace(interactions=SimpleNamespace(create=Mock(return_value=stream, side_effect=error)), close=Mock())
    return client, Mock(return_value=client), stream


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(speech, "_make_player", Mock(return_value=None))
    monkeypatch.setattr(speech, "play_wav", Mock(side_effect=AssertionError("Real playback is forbidden")))
    def deny(*args, **kwargs):
        pytest.fail("A speech test attempted external networking")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


def test_constructing_or_disallowed_cloud_speech_never_reads_key_or_creates_client(monkeypatch):
    environment = SimpleNamespace(get=Mock(side_effect=AssertionError("Key read before enablement")))
    monkeypatch.setattr(speech, "os", SimpleNamespace(environ=environment))
    _, factory, _ = client_for()
    engine = speech.GeminiSpeech(SpeechSettings(), client_factory=factory)
    assert engine.speak("Test.").status == "failed"
    factory.assert_not_called()
    environment.get.assert_not_called()


def test_stream_request_is_bounded_stateless_and_exports_pcm_wav(tmp_path):
    client, factory, stream = client_for()
    settings = SpeechSettings(timeout_s=4.25, gemini_voice="Kore")
    destination = tmp_path / "voice.wav"
    result = speech.GeminiSpeech(settings, cloud_enabled=True, client_factory=factory).speak("A fixed test.", output_path=destination)
    assert result.status == "completed" and result.mode == "wav" and result.backend == "gemini"
    assert result.voice == "Kore" and result.model == "gemini-3.8-flash-lite-tts"
    factory.assert_called_once_with(api_key=KEY, http_options={"timeout": 4250, "retry_options": {"attempts": 0}})
    request = client.interactions.create.call_args.kwargs
    assert request["stream"] is True and request["store"] is False and request["background"] is False
    assert request["automatic_continuation"] is False and request["tools"] == [] and request["timeout"] == 4.25
    assert request["response_format"] == {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000}
    assert request["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    assert "previous_interaction_id" not in request
    assert request["input"][0]["content"][0]["text"] == "A fixed test."
    with wave.open(str(destination), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
        assert audio.readframes(audio.getnframes()) == PCM
    client.close.assert_called_once_with()
    stream.close.assert_called_once_with()
    speech.play_wav.assert_not_called()


@pytest.mark.parametrize("events", [
    [delta()], [delta(), complete("incomplete")], [complete()],
    [delta(data=b"\x00"), complete()],
    [delta(data=b"", uri="https://do-not-fetch.invalid/audio"), complete()],
    [delta(mime_type="audio/wav"), complete()], [delta(sample_rate=16000), complete()],
    [delta(channels=2), complete()], [delta(channels=True), complete()],
    [delta(sample_rate=24000.0), complete()], [delta(data=b"", type="text"), complete()],
    [delta(**{"data": "not base64?"}), complete()],
])
def test_invalid_or_incomplete_audio_never_replaces_existing_export(tmp_path, events):
    client, factory, _ = client_for(events)
    target = tmp_path / "voice.wav"
    target.write_bytes(b"previous output")
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("Test.", output_path=target)
    assert result.status == "failed" and target.read_bytes() == b"previous output"
    client.close.assert_called_once_with()


def test_streaming_plays_chunks_before_completion_and_closes_player():
    player = SimpleNamespace(write=Mock(), finish=Mock(), stop=Mock(), close=Mock())
    def events():
        yield delta()
        player.write.assert_called_once_with(PCM)
        yield complete()
    client, factory, _ = client_for(events())
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory,
                                 player_factory=lambda *_: player).speak("Test.")
    assert result.status == "completed"
    player.finish.assert_called_once_with()
    player.close.assert_called_once_with()


def test_cancellation_stops_playback_and_discards_late_audio():
    player = SimpleNamespace(write=Mock(), finish=Mock(), stop=Mock(), close=Mock())
    engine = None
    def events():
        yield delta()
        engine.stop()
        yield delta()
        yield complete()
    client, factory, _ = client_for(events())
    engine = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory,
                                 player_factory=lambda *_: player)
    result = engine.speak("Test.")
    assert result.status == "cancelled"
    player.write.assert_called_once_with(PCM)
    player.stop.assert_called_once_with()
    player.finish.assert_not_called()
    player.close.assert_called_once_with()


def test_precancelled_and_closed_sessions_never_create_client():
    _, factory, _ = client_for()
    event = threading.Event()
    event.set()
    engine = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory)
    assert engine.speak("Test.", cancel=event).status == "cancelled"
    engine.close()
    assert engine.speak("Test.").status == "failed"
    factory.assert_not_called()


@pytest.mark.parametrize("text", ["", "   ", None, "x" * 4097, KEY])
def test_invalid_or_credential_text_is_not_uploaded(text):
    _, factory, _ = client_for()
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak(text)
    assert result.status == "failed"
    factory.assert_not_called()
    if text == KEY:
        assert result.text == ""


def test_remote_failure_is_sanitized_and_not_retried(tmp_path, caplog):
    client, factory, _ = client_for(error=RuntimeError("PRIVATE " + KEY))
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("Test.", output_path=tmp_path / "voice.wav")
    assert result.status == "failed" and KEY not in result.error + caplog.text
    assert "PRIVATE" not in result.error + caplog.text
    client.interactions.create.assert_called_once()
    client.close.assert_called_once()


def test_literal_tags_cannot_become_speech_directives(tmp_path):
    client, factory, _ = client_for()
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("<laugh> literal", output_path=tmp_path / "voice.wav")
    assert result.status == "completed"
    sent = client.interactions.create.call_args.kwargs["input"][0]["content"][0]["text"]
    assert "<laugh>" not in sent and "less than" in sent and "greater than" in sent


def test_audio_size_and_event_count_limits_preserve_output(tmp_path, monkeypatch):
    monkeypatch.setattr(speech, "MAX_PCM_BYTES", 8)
    _, factory, _ = client_for([delta(b"\x00\x01" * 5), complete()])
    assert speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("Test.", output_path=tmp_path / "voice.wav").status == "failed"
    assert not (tmp_path / "voice.wav").exists()
    monkeypatch.setattr(speech, "MAX_EVENTS", 1)
    _, factory, _ = client_for([delta(b"\x00\x01"), complete()])
    assert speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("Test.", output_path=tmp_path / "voice.wav").status == "failed"


def test_complete_wav_playback_is_private_and_removed_on_nonstreaming_local_player(monkeypatch):
    paths = []
    def play(path, **kwargs):
        paths.append(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with wave.open(str(path), "rb") as audio:
            assert audio.readframes(audio.getnframes()) == PCM
    monkeypatch.setattr(speech, "play_wav", play)
    _, factory, _ = client_for()
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("Test.")
    assert result.status == "completed" and len(paths) == 1 and not paths[0].exists()


def test_installed_sdk_serializes_streaming_tts_with_only_mock_transport(monkeypatch, tmp_path, caplog):
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    caplog.set_level("DEBUG")
    requests = []
    events = [delta(), complete()]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    def factory(*, api_key, http_options):
        return scene._default_client_factory(api_key=api_key, http_options={
            **http_options, "base_url": "https://offline-tts.invalid",
            "client_args": {"transport": httpx.MockTransport(respond), "trust_env": False},
        })
    result = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory).speak("Fixed audition.", output_path=tmp_path / "voice.wav")
    assert result.status == "completed", result.error
    assert len(requests) == 1 and requests[0].url.host == "offline-tts.invalid"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "gemini-3.8-flash-lite-tts" and payload["stream"] is True
    assert payload["input"][0]["content"][0]["annotations"][0]["type"] == "speech_metadata"
    assert payload["generation_config"]["speech_config"] == [{"voice": "Charon"}]
    assert payload["response_format"]["mime_type"] == "audio/l16"
    assert KEY not in caplog.text


def test_cancelled_speech_releases_busy_lock_with_bounded_pending_cleanup(tmp_path):
    release = threading.Event()
    client, factory, stream = client_for()
    stream.close.side_effect = lambda: release.wait(3)
    client.close.side_effect = lambda: release.wait(3)
    engine = speech.GeminiSpeech(SpeechSettings(), cloud_enabled=True, client_factory=factory)

    def events():
        engine.stop()
        yield delta()

    stream.events = events()
    started = time.monotonic()
    try:
        assert engine.speak("Cancelled.", output_path=tmp_path / "cancelled.wav").status == "cancelled"
        assert time.monotonic() - started < 1
        assert engine._busy.acquire(blocking=False)
        engine._busy.release()
        before = factory.call_count
        assert engine.speak("Do not accumulate cleanup work.").status == "failed"
        assert factory.call_count == before
        assert len(engine._cleanup_threads) == 2
        stream.close.assert_called_once()
        client.close.assert_called_once()
        assert not (tmp_path / "cancelled.wav").exists()
    finally:
        release.set()
        engine._release_transports((), wait_s=.5)
        engine.close()


def test_pcm_constructor_rolls_back_child_when_writer_cannot_start(monkeypatch):
    process = SimpleNamespace(pid=721338, stdin=io.BytesIO(), wait=Mock(return_value=0))
    popen, kill = Mock(return_value=process), Mock()
    monkeypatch.setattr(speech.subprocess, "Popen", popen)
    monkeypatch.setattr(speech.os, "killpg", kill)
    monkeypatch.setattr(speech.threading.Thread, "start", Mock(side_effect=RuntimeError("thread unavailable")))
    with pytest.raises(RuntimeError, match="thread unavailable"):
        speech._PCMPlayer("/fake/paplay", time.monotonic() + 1, threading.Event())
    assert popen.call_args.args[0] == ["/fake/paplay", "--raw", "--format=s16le", "--rate=24000", "--channels=1"]
    assert popen.call_args.kwargs["shell"] is False
    kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=.2)
    assert process.stdin.closed
