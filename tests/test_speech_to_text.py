"""Short command audio is tested with generated PCM and offline service clients."""

import base64
import io
import json
import socket
import struct
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from vidi.ai import scene_description as scene
from vidi.audio import speech_to_text as stt
from vidi.core.configuration import CloudSettings, VoiceSettings
from vidi.core.local_process import LocalProcessCancelled, LocalProcessError


TEST_KEY = "offline-command-test-key-never-a-real-credential"
PRIVATE = "private-device-or-service-diagnostic"


@pytest.fixture(autouse=True)
def isolate_external_effects(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", TEST_KEY)
    monkeypatch.delenv("PULSE_SOURCE", raising=False)
    monkeypatch.setattr(stt, "run_local_process", Mock(side_effect=AssertionError("Real audio input is forbidden")))

    def deny_network(*args, **kwargs):
        pytest.fail("Audio tests must never open external connections")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)


def wav_bytes(*, rate=16000, channels=1, seconds=.1, width=2, silent=False):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        sample = bytes(width) if silent else b"\x01" + bytes(width - 1)
        audio.writeframes(sample * int(rate * seconds) * channels)
    return buffer.getvalue()


def clip(**kwargs):
    data = wav_bytes(**kwargs)
    return stt.AudioClip(data, kwargs.get("seconds", .1))


def fake_client(*, text='{"text":"read this sign"}', status="completed", error=None, response=None):
    if response is None:
        response = SimpleNamespace(status=status, output_text=text)
    client = SimpleNamespace(
        interactions=SimpleNamespace(create=Mock(return_value=response, side_effect=error)),
        files=SimpleNamespace(upload=Mock(side_effect=AssertionError("Files uploads are forbidden"))),
        close=Mock(),
    )
    return client, Mock(return_value=client)


def transcriber(factory, **voice_options):
    return stt.GeminiCommandTranscriber(CloudSettings(), VoiceSettings(**voice_options), client_factory=factory)


def process_result(*, stdout=b"", returncode=0):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=f"{PRIVATE} {TEST_KEY}".encode())


def test_construction_does_not_open_audio_or_read_credentials(monkeypatch):
    environment = SimpleNamespace(get=Mock(side_effect=AssertionError("Credential read during construction")))
    monkeypatch.setattr(stt, "os", SimpleNamespace(environ=environment))
    factory = Mock(side_effect=AssertionError("Client created during construction"))
    stt.MicrophoneRecorder(VoiceSettings())
    stt.GeminiCommandTranscriber(CloudSettings(), VoiceSettings(), client_factory=factory)
    environment.get.assert_not_called()
    factory.assert_not_called()
    stt.run_local_process.assert_not_called()


def test_capture_uses_fixed_raw_pcm_arguments_and_forwards_cancellation(monkeypatch):
    event = threading.Event()
    pcm = b"\x01\x00" * stt.SAMPLE_RATE * 2
    run = Mock(return_value=process_result(stdout=pcm))
    monkeypatch.setattr(stt, "run_local_process", run)
    settings = VoiceSettings(device="plughw:CARD=USB,DEV=0", record_seconds=2)
    result = stt.MicrophoneRecorder(settings).record(cancel_event=event)
    run.assert_called_once_with(
        ["arecord", "--quiet", "--device", settings.device, "--file-type", "raw",
         "--format", "S16_LE", "--rate", "16000", "--channels", "1", "--duration", "2"],
        input_bytes=b"", timeout_s=5.0, cancel_event=event,
    )
    with wave.open(io.BytesIO(result.wav_bytes), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        assert audio.readframes(audio.getnframes()) == pcm
    assert result.duration_s == 2.0
    assert "wav_bytes" not in repr(result)


def test_default_pulse_source_is_verified_before_recording(monkeypatch):
    pcm = b"\x01\x00" * stt.SAMPLE_RATE
    run = Mock(side_effect=[process_result(stdout=b"alsa_input.usb_headset\n"), process_result(stdout=pcm)])
    monkeypatch.setattr(stt, "run_local_process", run)
    result = stt.MicrophoneRecorder(VoiceSettings(record_seconds=1)).record()
    assert result.duration_s == 1
    assert run.call_args_list[0].args[0] == ["pactl", "get-default-source"]
    assert run.call_args_list[0].kwargs["timeout_s"] == 2.0
    assert run.call_args_list[1].args[0][0] == "arecord"


@pytest.mark.parametrize("source", [b"alsa_output.hdmi.monitor\n", b"", b"one\ntwo", b"source\x00bad"])
def test_unusable_default_or_output_monitor_never_opens_recorder(monkeypatch, source):
    run = Mock(return_value=process_result(stdout=source))
    monkeypatch.setattr(stt, "run_local_process", run)
    with pytest.raises(stt.SpeechRecognitionError, match="default input|output monitor"):
        stt.MicrophoneRecorder(VoiceSettings()).record()
    assert run.call_count == 1
    assert run.call_args.args[0] == ["pactl", "get-default-source"]


def test_explicit_pulse_source_avoids_probe_and_still_rejects_monitor(monkeypatch):
    monkeypatch.setenv("PULSE_SOURCE", "alsa_output.speaker.monitor")
    with pytest.raises(stt.SpeechRecognitionError, match="output monitor"):
        stt.MicrophoneRecorder(VoiceSettings()).record()
    stt.run_local_process.assert_not_called()


@pytest.mark.parametrize("failure", ["partial", "oversized", "nonzero", "missing", "timeout"])
def test_capture_failures_are_sanitized_and_never_return_partial_audio(monkeypatch, failure):
    full_pcm = b"\x01\x00" * stt.SAMPLE_RATE
    if failure in ("missing", "timeout"):
        run = Mock(side_effect=LocalProcessError(f"{PRIVATE} {TEST_KEY}"))
    else:
        payload = full_pcm[:-2] if failure == "partial" else full_pcm + b"\x01\x00" if failure == "oversized" else full_pcm
        run = Mock(return_value=process_result(stdout=payload, returncode=1 if failure == "nonzero" else 0))
    monkeypatch.setattr(stt, "run_local_process", run)
    with pytest.raises(stt.SpeechRecognitionError) as caught:
        stt.MicrophoneRecorder(VoiceSettings(device="hw:test", record_seconds=1)).record()
    assert TEST_KEY not in str(caught.value) and PRIVATE not in str(caught.value)


def test_capture_cancellation_before_and_after_process_discards_audio(monkeypatch):
    event = threading.Event()
    event.set()
    recorder = stt.MicrophoneRecorder(VoiceSettings(device="hw:test", record_seconds=1))
    with pytest.raises(LocalProcessCancelled):
        recorder.record(cancel_event=event)
    stt.run_local_process.assert_not_called()
    event.clear()

    def completed(*args, **kwargs):
        event.set()
        return process_result(stdout=b"\x01\x00" * stt.SAMPLE_RATE)

    monkeypatch.setattr(stt, "run_local_process", completed)
    with pytest.raises(LocalProcessCancelled):
        recorder.record(cancel_event=event)


def test_running_capture_cancellation_is_preserved(monkeypatch):
    monkeypatch.setattr(stt, "run_local_process", Mock(side_effect=LocalProcessCancelled("cancelled")))
    with pytest.raises(LocalProcessCancelled):
        stt.MicrophoneRecorder(VoiceSettings(device="hw:test")).record()


@pytest.mark.parametrize("rate,channels,seconds", [(8000, 1, .1), (16000, 1, 5), (48000, 2, 5)])
def test_replay_accepts_complete_bounded_16_bit_mono_or_stereo(tmp_path, rate, channels, seconds):
    data = wav_bytes(rate=rate, channels=channels, seconds=seconds)
    source = tmp_path / "command.wav"
    source.write_bytes(data)
    result = stt.MicrophoneRecorder(VoiceSettings(audio_path=source)).record()
    assert result.duration_s == seconds and result.sample_rate == rate and result.channels == channels
    assert result.wav_bytes == data
    stt.run_local_process.assert_not_called()


@pytest.mark.parametrize("kind", ["empty", "not_wav", "truncated", "too_large", "too_long", "8_bit", "3_channels", "low_rate", "high_rate"])
def test_invalid_replay_audio_is_rejected_without_recording_or_network(tmp_path, kind):
    data = {
        "empty": b"", "not_wav": b"not a wave", "truncated": wav_bytes()[:-2],
        "too_large": b"x" * (stt.MAX_AUDIO_BYTES + 1), "too_long": wav_bytes(seconds=5.01),
        "8_bit": wav_bytes(width=1), "3_channels": wav_bytes(channels=3),
        "low_rate": wav_bytes(rate=7999), "high_rate": wav_bytes(rate=48001),
    }[kind]
    source = tmp_path / "invalid.wav"
    source.write_bytes(data)
    with pytest.raises(stt.SpeechRecognitionError):
        stt.MicrophoneRecorder(VoiceSettings(audio_path=source)).record()
    stt.run_local_process.assert_not_called()


def test_missing_replay_path_has_sanitized_error(tmp_path):
    source = tmp_path / f"{PRIVATE}.wav"
    with pytest.raises(stt.SpeechRecognitionError) as caught:
        stt.MicrophoneRecorder(VoiceSettings(audio_path=source)).record()
    assert PRIVATE not in str(caught.value)


def test_replay_and_upload_strip_ancillary_metadata(tmp_path):
    original = wav_bytes()
    metadata = b"PRIVATE_AUDIO_FILE_METADATA"
    chunk = b"JUNK" + struct.pack("<I", len(metadata)) + metadata + (b"\x00" if len(metadata) % 2 else b"")
    annotated = bytearray(original[:12] + chunk + original[12:])
    struct.pack_into("<I", annotated, 4, len(annotated) - 8)
    source = tmp_path / "annotated.wav"
    source.write_bytes(annotated)
    captured = stt.MicrophoneRecorder(VoiceSettings(audio_path=source)).record()
    assert captured.wav_bytes == original
    client, factory = fake_client()
    result = transcriber(factory).transcribe(stt.AudioClip(bytes(annotated), .1))
    assert result.status == "ready"
    sent = base64.b64decode(client.interactions.create.call_args.kwargs["input"][1]["data"], validate=True)
    assert sent == original and metadata not in sent
    assert source.read_bytes() == bytes(annotated)


def test_transcription_request_is_inline_stateless_bounded_and_preserves_negation():
    client, factory = fake_client(text='{"text":"  do not stop\\n speaking  "}')
    audio = clip()
    result = transcriber(factory, model="chosen-audio-model", timeout_s=4.25).transcribe(audio)
    assert result.status == "ready" and result.text == "do not stop speaking"
    assert result.model == "chosen-audio-model" and result.elapsed_s >= 0
    assert "do not stop speaking" not in repr(result)
    factory.assert_called_once_with(api_key=TEST_KEY, http_options={"timeout": 4250, "retry_options": {"attempts": 0}})
    request = client.interactions.create.call_args.kwargs
    assert request["model"] == "chosen-audio-model" and request["timeout"] == 4.25
    assert request["input"][1] == {"type": "audio", "data": base64.b64encode(audio.wav_bytes).decode("ascii"), "mime_type": "audio/wav"}
    assert request["tools"] == [] and request["store"] is False and request["stream"] is False
    assert request["background"] is False and request["automatic_continuation"] is False
    assert request["generation_config"] == {"max_output_tokens": 256, "tool_choice": "none"}
    assert request["response_format"]["mime_type"] == "application/json"
    assert request["response_format"]["schema"]["additionalProperties"] is False
    assert "previous_interaction_id" not in request and "uri" not in request["input"][1]
    assert "Preserve negations" in request["system_instruction"]
    client.files.upload.assert_not_called()
    client.close.assert_called_once_with()


def test_silent_pcm_never_constructs_client_or_reads_key(monkeypatch):
    client, factory = fake_client()
    environment = SimpleNamespace(get=Mock(side_effect=AssertionError("Key read for silent PCM")))
    monkeypatch.setattr(stt, "os", SimpleNamespace(environ=environment))
    result = transcriber(factory).transcribe(clip(silent=True))
    assert result.status == "no_speech" and result.text == ""
    factory.assert_not_called()
    environment.get.assert_not_called()


def test_missing_key_fails_before_client_creation(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "different-key-must-not-be-used")
    _, factory = fake_client()
    result = transcriber(factory).transcribe(clip())
    assert result.status == "failed" and "key" in result.error
    factory.assert_not_called()


@pytest.mark.parametrize("invalid", [None, b"not-a-clip", stt.AudioClip(b"invalid", .1), stt.AudioClip(wav_bytes(seconds=5.01), .1)])
def test_input_audio_is_validated_from_bytes_before_creating_client(invalid):
    _, factory = fake_client()
    result = transcriber(factory).transcribe(invalid)
    assert result.status == "failed" and not result.text
    factory.assert_not_called()


def test_cancellation_before_validation_or_client_prevents_network():
    event = threading.Event()
    event.set()
    _, factory = fake_client()
    result = transcriber(factory).transcribe(None, cancel_event=event)
    assert result.status == "cancelled"
    factory.assert_not_called()


@pytest.mark.parametrize("stage", ["client", "request"])
def test_cancellation_at_service_boundaries_discards_late_transcripts_and_closes_client(stage):
    event = threading.Event()
    client, factory = fake_client()
    if stage == "client":
        def create_client(**kwargs):
            event.set()
            return client
        factory.side_effect = create_client
    else:
        def create_response(**kwargs):
            event.set()
            return SimpleNamespace(status="completed", output_text='{"text":"describe the scene"}')
        client.interactions.create.side_effect = create_response
    result = transcriber(factory).transcribe(clip(), cancel_event=event)
    assert result.status == "cancelled" and result.text == ""
    assert client.interactions.create.call_count == (0 if stage == "client" else 1)
    client.close.assert_called_once_with()


@pytest.mark.parametrize("text", [
    'not json', '{"text":"stop","text":"describe"}', '{"text":"stop","action":"describe"}',
    '{"text":3}', '[]', json.dumps({"text": "x" * 501}), 'x' * 4097,
    json.dumps({"text": f"secret {TEST_KEY.upper()}"}), json.dumps({"text": "stop\x00"}),
])
def test_invalid_or_secret_transcripts_fail_without_leaking_text(text, caplog):
    client, factory = fake_client(text=text)
    result = transcriber(factory).transcribe(clip())
    assert result.status == "failed" and not result.text
    assert TEST_KEY.casefold() not in (result.error + caplog.text).casefold()
    assert text not in (result.error, caplog.text)
    client.close.assert_called_once_with()


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", "requires_continuation"])
def test_unfinished_service_results_never_become_commands(status):
    client, factory = fake_client(status=status)
    result = transcriber(factory).transcribe(clip())
    assert result.status == "failed" and result.text == ""
    client.close.assert_called_once_with()


@pytest.mark.parametrize("status_field", [{}, {"status": None}, {"status": "in_progress"}])
def test_missing_or_noncompleted_status_cannot_dispatch_a_transcribed_command(status_field):
    response = {**status_field, "output_text": '{"text":"read this sign"}'}
    client, factory = fake_client(response=response)
    result = transcriber(factory).transcribe(clip())
    assert result.status == "failed" and result.text == ""
    client.close.assert_called_once_with()


def test_empty_transcription_and_nontext_steps_are_handled_without_command():
    client, factory = fake_client(text='{"text":"  "}')
    assert transcriber(factory).transcribe(clip()).status == "no_speech"
    client, factory = fake_client(response={"status": "completed", "steps": [
        {"type": "thought", "content": [{"type": "text", "text": '{"text":"stop"}'}]},
        {"type": "function_call", "text": '{"text":"stop"}'},
    ]})
    assert transcriber(factory).transcribe(clip()).status == "failed"
    client.close.assert_called_once_with()


@pytest.mark.parametrize("error", [TimeoutError(f"{PRIVATE} {TEST_KEY}"), ConnectionError(f"{PRIVATE} {TEST_KEY}"), RuntimeError(f"{PRIVATE} {TEST_KEY}")])
def test_service_failures_are_sanitized_without_retries_and_client_is_closed(error, caplog):
    client, factory = fake_client(error=error)
    result = transcriber(factory).transcribe(clip())
    assert result.status == "failed" and not result.text
    assert PRIVATE not in result.error + caplog.text and TEST_KEY not in result.error + caplog.text
    assert client.interactions.create.call_count == 1
    client.close.assert_called_once_with()


def test_factory_and_cleanup_errors_cannot_leak_private_diagnostics():
    factory = Mock(side_effect=RuntimeError(f"{PRIVATE} {TEST_KEY}"))
    failed = transcriber(factory).transcribe(clip())
    assert failed.status == "failed" and PRIVATE not in failed.error and TEST_KEY not in failed.error
    client, factory = fake_client()
    client.close.side_effect = RuntimeError(f"{PRIVATE} {TEST_KEY}")
    assert transcriber(factory).transcribe(clip()).status == "ready"


@pytest.mark.parametrize("status_code", [200, 429, 503])
def test_installed_sdk_serializes_audio_using_only_offline_mock_transport(monkeypatch, caplog, status_code):
    pytest.importorskip("google.genai", reason="Optional Gemini SDK is not installed")
    httpx = pytest.importorskip("httpx", reason="Optional SDK compatibility requires httpx")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx")
    caplog.set_level("DEBUG")
    requests = []
    audio = clip()
    payload = {"id": "offline-audio-interaction", "object": "interaction", "model": "chosen-audio-model",
               "status": "completed", "created": "2026-11-15T12:00:00Z", "updated": "2026-11-15T12:00:00Z",
               "steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"text":"read this sign"}'}]}]}

    def respond(request):
        requests.append(request)
        return httpx.Response(status_code, json=payload if status_code == 200 else {
            "error": {"code": "resource_exhausted" if status_code == 429 else "unavailable",
                      "message": f"{PRIVATE} {TEST_KEY}"}})

    def factory(*, api_key, http_options):
        assert api_key == TEST_KEY and http_options == {"timeout": 3000, "retry_options": {"attempts": 0}}
        return scene._default_client_factory(api_key=api_key, http_options={
            **http_options, "base_url": "https://offline-command.invalid",
            "client_args": {"transport": httpx.MockTransport(respond), "trust_env": False},
        })

    result = transcriber(factory, timeout_s=3.0, model="chosen-audio-model").transcribe(audio)
    assert result.status == ("ready" if status_code == 200 else "failed")
    assert len(requests) == 1  # Includes the SDK's retry layer, not merely our fake factory.
    request = requests[0]
    assert request.method == "POST" and request.url.host == "offline-command.invalid"
    assert request.url.path.endswith("/interactions")
    assert request.extensions["timeout"]["read"] == 3.0
    body = json.loads(request.content)
    assert body["model"] == "chosen-audio-model"
    assert body["store"] is False and body["stream"] is False and body["background"] is False
    assert body["tools"] == [] and "previous_interaction_id" not in body
    assert body["generation_config"] == {"max_output_tokens": 256, "tool_choice": "none"}
    assert body["response_format"]["schema"]["additionalProperties"] is False
    content = body["input"][0]["content"]
    assert content[1]["type"] == "audio" and content[1]["mime_type"] == "audio/wav"
    assert "uri" not in content[1] and base64.b64decode(content[1]["data"], validate=True) == audio.wav_bytes
    assert TEST_KEY not in caplog.text and PRIVATE not in caplog.text
    assert content[1]["data"] not in caplog.text
