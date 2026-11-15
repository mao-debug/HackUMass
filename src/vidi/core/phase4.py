"""Explicit scene snapshots and independent local proximity feedback in a live UI.

The camera never waits for cloud, serial, or speech work. D submits one owned
image; P is a local action. There is no automatic upload or speech queue.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, replace
import logging
import math
import threading
import time
import textwrap
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vidi.ai.scene_description import GeminiSceneDescriber, SceneDescriptionError
from vidi.audio.text_to_speech import LocalSpeech
from vidi.camera.capture import CaptureError
from vidi.core.types import FramePacket, OcrResult, SpeechResult

if TYPE_CHECKING:
    from vidi.core.configuration import AppSettings
    from vidi.core.controller import SimulationResult

logger = logging.getLogger(__name__)
WINDOW_TITLE = "vidi: D describe | P proximity | S stop speech | R retry | Q quit"
ACTION_LIMIT = 120
VOICE_WINDOW_TITLE = "vidi: M command | D describe | T read | P proximity | H help | S stop | R retry | Q quit"
HELP_TEXT = "Press M and say describe, read text, proximity, stop, or help. Keyboard: D describe, T read text, P proximity, S stop, R retry, Q quit."
OFFLINE_WINDOW_TITLE = "vidi OFFLINE: P proximity | H help | S stop | R retry sensors | Q quit"
OFFLINE_HELP_TEXT = "Offline mode. P requests local proximity if enabled. H repeats help. S stops speech. R retries local sensors. Q or Escape quits. Camera descriptions, text reading, and microphone commands are disabled."


def _record_voice(settings, cancel):
    from vidi.audio.speech_to_text import MicrophoneRecorder
    return MicrophoneRecorder(settings.voice).record(cancel_event=cancel)


def _transcribe_voice(settings, clip, cancel):
    from vidi.audio.speech_to_text import GeminiCommandTranscriber
    return GeminiCommandTranscriber(settings.cloud, settings.voice).transcribe(clip, cancel_event=cancel)


def _parse_voice_command(text):
    from vidi.assistant.command_handler import parse_command
    return parse_command(text)


def _recognize_text(settings, packet, cancel):
    from vidi.ai.gemini_text_recognition import GeminiTextRecognizer
    if cancel.is_set():
        return None
    recognizer = GeminiTextRecognizer(settings.cloud, settings.ocr)
    if cancel.is_set():
        return None
    return recognizer.recognize(packet, cancel_event=cancel)


def compose_feedback(description: str | None, proximity_text: str | None,
                     image_age_s: float, max_words: int) -> str:
    """Keep sensor feedback first and bound the entire utterance, visual copy first.

    Sensor measurements and camera observations remain separate statements.
    The original model description belongs in the report, never this shortened
    speech copy. Very small budgets use an unknown-status fallback rather than
    truncating a sensor sentence into an apparent measurement.
    """
    if type(max_words) is not int or max_words < 1:
        raise ValueError("max_words must be a positive integer.")
    if not isinstance(image_age_s, (int, float)) or not math.isfinite(image_age_s) or image_age_s < 0:
        raise ValueError("image_age_s must be finite and nonnegative.")
    sensor = " ".join((proximity_text or "").split())
    visual = " ".join((description or "").split())
    sensor_words = sensor.split()
    if len(sensor_words) > max_words:
        fallback = "Near-field sensor details unavailable."
        if max_words < len(fallback.split()):
            fallback = "Sensors unknown." if max_words >= 2 else "Unknown."
        return fallback
    remaining = max_words - len(sensor_words)
    prefix = f"Camera snapshot from {image_age_s:.1f} seconds ago:"
    visual_words = (prefix + " " + visual).split() if visual else []
    if len(visual_words) > remaining:
        # Do not speak an age caption with no remaining visual observation.
        if remaining <= len(prefix.split()):
            visual_words = []
        else:
            visual_words = visual_words[:remaining]
            visual_words[-1] = visual_words[-1].rstrip(".,;:") + "…"
    return " ".join(sensor_words + visual_words)


class _Task:
    """One daemon job, with cancellation and no queue or unbounded shutdown."""

    def __init__(self, operation):
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.value = None
        self.failed = False
        self.failure = None

        def run():
            try:
                if not self.cancel.is_set():
                    self.value = operation(self.cancel)
            except BaseException as error:
                # Workers never emit raw service, serial, or engine exceptions.
                self.failed = True
                self.failure = error
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True, name="vidi-phase4-action")
        self.thread.start()

    def close(self):
        self.cancel.set()
        try:
            self.thread.join(timeout=0.5)
        except KeyboardInterrupt:
            # A repeated interrupt must not skip cleanup of the other worker.
            pass


def _open_proximity(settings):
    from vidi.sensors.proximity import ProximityMonitor

    return ProximityMonitor(settings.proximity)


def _format_proximity(snapshot, settings) -> str:
    from vidi.sensors.proximity import format_proximity

    return format_proximity(snapshot, nearby_mm=settings.proximity.nearby_mm)


def _refresh_proximity(snapshot, settings):
    from vidi.sensors.proximity import refresh_snapshot

    return refresh_snapshot(snapshot, settings.proximity.stale_after_s)


def _snapshot_dict(snapshot):
    return asdict(snapshot) if snapshot is not None else None


def render_phase4_preview(packet: FramePacket, described: FramePacket | None, description: dict, *, proximity_text: str,
                          cloud_busy: bool, audio_busy: bool, paused: bool,
                          voice_status: str | None = None, voice_error: str | None = None,
                          offline: bool = False):
    """Show current pixels separately from the exact image submitted with D."""
    live = packet.image_bgr.copy()
    analysis = np.zeros_like(live) if described is None else described.image_bgr.copy()
    if analysis.shape != live.shape:
        analysis = cv2.resize(analysis, (live.shape[1], live.shape[0]))

    def lines(image, texts, *, top=0):
        width = image.shape[1]
        scale = min(0.5, max(0.2, width / 1200))
        texts = [line for text in texts for line in textwrap.wrap(text, max(20, width // 9))]
        height = min(image.shape[0], len(texts) * 20 + 8)
        cv2.rectangle(image, (0, top), (width, min(image.shape[0], top + height)), (25, 25, 25), -1)
        for index, text in enumerate(texts):
            y = top + 18 + index * 20
            if y >= image.shape[0]:
                break
            cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)

    controls = "M command / D describe / T read / P proximity / H help / S stop / R retry / Q quit" if voice_status is not None else "D describe / P proximity / S stop / R retry / Q quit"
    if offline:
        controls = "OFFLINE: P proximity / H help / S stop / R retry sensors / Q quit"
    texts = [f"LIVE CAMERA frame {packet.frame_id}", proximity_text,
             "Speech active" if audio_busy else controls]
    if voice_status is not None:
        texts.append("Voice: " + voice_status.replace("_", " "))
        if voice_error:
            texts.append(voice_error)
    lines(live, texts)
    status = description["status"]
    title = "Press D for one scene snapshot"
    if offline:
        title = "OFFLINE - cloud uploads and microphone disabled"
    elif cloud_busy:
        title = ("TEXT READING" if description.get("action") == "read_text" else "SCENE REQUEST") + " PENDING - camera continues"
    elif paused:
        title = "SCENE UNAVAILABLE - press R to rearm"
    elif described is not None:
        age = max(0.0, (time.monotonic_ns() - described.captured_monotonic_ns) / 1e9)
        title = f"{'TEXT SNAPSHOT' if description.get('action') == 'read_text' else 'SNAPSHOT'} frame {described.frame_id}, {age:.1f}s old - {status}"
    lines(analysis, [title, "Press H for local help." if offline else description.get("text") or description.get("error") or "No scene description requested."])
    return cv2.hconcat((live, analysis))


def run_phase4_live(settings: AppSettings, *, proximity: bool, speak: bool, open_source,
                    question: str = "What is in front of me?", voice_commands: bool = False,
                    offline: bool = False) -> SimulationResult:
    """Run requested scene/proximity actions, optionally adding M/T/H controls.

    S invalidates pending narration and cancels local audio. An already started
    HTTP request may finish, but its response cannot start cancelled speech.
    Cloud failures pause D until R; local P feedback remains independent.
    """
    # Import inside the entry point: controller also routes into this module.
    from vidi.core.controller import (
        SimulationError, SimulationResult, _observed_rate, _save_image, _save_report,
    )

    if not settings.simulation.preview:
        raise SimulationError("Scene live mode requires the desktop preview.")
    if offline and voice_commands:
        raise SimulationError("Offline mode cannot enable voice commands or microphone capture.")
    if settings.models.object_detection_backend not in (None, "none"):
        raise SimulationError("Scene live mode requires the object detector to be off.")
    if not isinstance(question, str) or not question.strip() or len(question) > 500:
        raise SimulationError("Ask a nonempty scene question of at most 500 characters.")
    if voice_commands and (not hasattr(settings, "voice") or settings.models.ocr_backend not in (None, "gemini")):
        raise SimulationError("Voice mode requires voice settings and the Gemini text-reading backend.")

    actions = deque(maxlen=ACTION_LIMIT)
    totals = {"actions": 0, "cloud_requests": 0, "local_reads": 0, "speech_attempts": 0, "ignored_actions": 0}
    description = {"status": "not_requested", "text": None, "error": None, "model": None,
                   "elapsed_s": None, "frame_id": None, "captured_monotonic_ns": None}
    speech_result = SpeechResult()
    cloud_task = local_task = None
    voice_task = None
    voice_clip = voice_pending = None
    voice_stage = None
    voice_generation = 0
    voice_paused = False
    voice_state = {"status": "not_requested", "command": None, "model": None,
                   "elapsed_s": None, "error": None}
    reading = OcrResult()
    cloud_action = "describe"
    local_stage = None
    cloud_generation = local_generation = generation = 0
    cloud_packet = described = last_packet = last_snapshot = None
    cloud_paused = interrupted = False
    cloud_started = None
    cooldown_until = 0.0
    count = 0
    first_frame_ns = None
    output = None
    started = processing_started = time.monotonic()
    processing_ended = None
    monitor = None
    last_proximity_text = None

    def record(kind, status, **fields):
        totals["actions"] += 1
        actions.append({"action": kind, "status": status, "monotonic_ns": time.monotonic_ns(), **fields})

    def checkpoint():
        if settings.simulation.report_path is None:
            return
        report = {
            "schema_version": 4 if voice_commands else 3,
            "mode": "phase5_voice_live" if voice_commands else "phase4_scene_live", "continuous": True,
            "source": settings.camera.source, "frames_processed": count,
            "width": last_packet.image_bgr.shape[1] if last_packet else None,
            "height": last_packet.image_bgr.shape[0] if last_packet else None,
            "startup_s": processing_started - started,
            "processing_s": (processing_ended if processing_ended is not None else time.monotonic()) - processing_started,
            "capture_fps": _observed_rate(count, first_frame_ns,
                                          last_packet.captured_monotonic_ns if last_packet else None),
            "last_preview_frame_id": last_packet.frame_id if last_packet else None,
            "description": dict(description), "speech": asdict(speech_result),
            "proximity": _snapshot_dict(last_snapshot), "proximity_enabled": proximity,
            "proximity_feedback": last_proximity_text,
            "cloud_uploads_paused": cloud_paused, "cloud_min_interval_s": max(5.0, settings.cloud.live_interval_s),
            "actions": list(actions), "action_totals": dict(totals),
            "action_records_limit": ACTION_LIMIT, "action_records_truncated": totals["actions"] > len(actions),
            "action_records_dropped": totals["actions"] - len(actions),
            "interrupted": interrupted, "output_path": str(output) if output else None,
            "output_frame_id": (cloud_packet or last_packet).frame_id if output else None,
            "report_path": str(settings.simulation.report_path),
        }
        if voice_commands:
            report.update(voice_commands_enabled=True, voice=dict(voice_state),
                          voice_paused=voice_paused, ocr=asdict(reading))
        if offline:
            report.update(offline=True, cloud_uploads_disabled=True)
        _save_report(settings.simulation.report_path, report)

    def read_sensors(cancel):
        if monitor is None:
            return None, ""
        try:
            snapshot = monitor.fresh(after_ns=time.monotonic_ns(), timeout_s=0.75, cancel_event=cancel)
            return snapshot, _format_proximity(snapshot, settings)
        except Exception:
            return None, "Near-field sensors unavailable; left and right are unknown."

    def start_local(kind):
        nonlocal local_task, local_stage, local_generation, speech_result
        local_generation = generation
        local_stage = kind
        speech_result = SpeechResult(status="preparing" if speak else "not_requested")
        record(kind, "pending")
        checkpoint()
        local_task = _Task(read_sensors)

    def cloud_operation(cancel):
        try:
            if offline or cancel.is_set():
                return None, None
            if cloud_action == "read_text":
                response = _recognize_text(settings, cloud_packet, cancel)
                return (None, None) if cancel.is_set() else (response, None)
            describer = GeminiSceneDescriber(settings.cloud)
            if cancel.is_set():
                return None, None
            response = describer.describe(cloud_packet.image_bgr, question, cancel_event=cancel) if voice_commands else describer.describe(cloud_packet.image_bgr, question)
            return response, None
        except SceneDescriptionError as error:
            if voice_commands and cancel.is_set():
                return None, None
            return None, str(error)
        except Exception:
            if voice_commands and cancel.is_set():
                return None, None
            return None, "The requested camera action failed. Check the connection and model access."

    def start_camera(action, packet):
        nonlocal generation, cloud_generation, cloud_packet, cloud_task, cloud_started, cloud_action, reading
        generation += 1
        cloud_generation = generation
        cloud_action = action
        cloud_packet = FramePacket(packet.frame_id, packet.captured_monotonic_ns, packet.image_bgr.copy())
        if action == "read_text":
            reading = OcrResult(status="pending", frame_id=packet.frame_id, backend="gemini", model=settings.cloud.model)
        else:
            description.update(status="pending", text=None, error=None, model=settings.cloud.model,
                               elapsed_s=None, frame_id=packet.frame_id,
                               captured_monotonic_ns=packet.captured_monotonic_ns)
        totals["cloud_requests"] += 1
        record(action, "pending", frame_id=packet.frame_id)
        checkpoint()
        cloud_started = time.monotonic()
        cloud_task = _Task(cloud_operation)

    def voice_busy():
        return voice_task is not None or voice_clip is not None or voice_pending is not None

    def cancel_voice():
        nonlocal voice_clip, voice_pending, voice_stage
        if voice_task is not None:
            voice_task.cancel.set()
        if voice_busy():
            voice_state.update(status="cancelled", error=None)
        voice_clip = voice_pending = None
        voice_stage = None

    def stop_narration():
        nonlocal generation, speech_result
        generation += 1
        if cloud_task is not None:
            cloud_task.cancel.set()
        if local_task is not None:
            local_task.cancel.set()
            speech_result = replace(speech_result, status="cancelled")
        cancel_voice()

    def start_help():
        nonlocal generation, local_task, local_stage, local_generation, speech_result
        generation += 1
        cancel_voice()
        record("help", "ready")
        help_text = OFFLINE_HELP_TEXT if offline else HELP_TEXT
        logger.info("Requested help: %s", help_text)
        if speak:
            speech_result = SpeechResult(status="pending", mode="playback", text=help_text)
            local_stage = "speech"
            local_generation = generation
            totals["speech_attempts"] += 1
            checkpoint()

            def operation(cancel):
                spoken = SpeechResult(status="cancelled", mode="playback", text=help_text) if cancel.is_set() else (
                    LocalSpeech(settings.speech, settings.models.text_to_speech_backend).speak(help_text, cancel_event=cancel))
                return last_snapshot, last_proximity_text, spoken

            local_task = _Task(operation)
        else:
            checkpoint()

    def advance_voice(packet):
        """Dispatch stages only AFTER this frame's S/Q/P input has been handled."""
        nonlocal voice_task, voice_clip, voice_stage, voice_pending, generation, voice_generation
        if not voice_commands or voice_generation != generation or voice_paused:
            return
        if voice_clip is not None:
            clip, voice_clip = voice_clip, None
            voice_stage = "transcribing"
            voice_state.update(status="transcribing")
            totals["cloud_requests"] += 1
            record("voice_transcribe", "pending")
            checkpoint()
            voice_task = _Task(lambda cancel, clip=clip: _transcribe_voice(settings, clip, cancel))
            voice_task.kind = "transcribing"
        elif voice_pending is not None:
            action = voice_pending
            if action in ("describe", "read_text"):
                if cloud_paused:
                    voice_pending = None
                    voice_state.update(status="failed", error="Camera requests are paused. Press R to retry.")
                    checkpoint()
                elif cloud_task is None and local_task is None and time.monotonic() >= cooldown_until:
                    voice_pending = None
                    voice_state.update(status="ready")
                    start_camera(action, packet)
            else:
                voice_pending = None
                voice_state.update(status="ready")
                if action == "stop":
                    stop_narration()
                    record("stop", "completed")
                    checkpoint()
                elif action == "help":
                    start_help()
                elif action == "proximity":
                    if proximity and local_task is None:
                        generation += 1
                        totals["local_reads"] += 1
                        start_local("proximity")
                    else:
                        record("proximity", "ignored_busy_or_disabled")
                        checkpoint()

    def speak_operation(cancel, snapshot, sensor_text, visual_text, captured_ns, unavailable):
        if snapshot is not None:
            try:
                snapshot = _refresh_proximity(snapshot, settings)
                sensor_text = _format_proximity(snapshot, settings)
            except Exception:
                snapshot = None
                sensor_text = "Near-field sensors unavailable; left and right are unknown."
        if unavailable:
            sensor_text += " Scene description unavailable."
        text = compose_feedback(
            visual_text, sensor_text,
            max(0.0, (time.monotonic_ns() - captured_ns) / 1e9) if captured_ns is not None else 0.0,
            settings.speech.max_words,
        )
        if cancel.is_set():
            return snapshot, sensor_text, SpeechResult(status="cancelled", mode="playback", text=text)
        spoken = LocalSpeech(settings.speech, settings.models.text_to_speech_backend).speak(
            text, cancel_event=cancel,
        )
        return snapshot, sensor_text, spoken

    def receive_speech(task):
        nonlocal last_snapshot, last_proximity_text, speech_result
        if task.failed or task.value is None:
            return False
        last_snapshot, last_proximity_text, speech_result = task.value
        return True

    def feedback_image(stage):
        if stage == "scene_feedback":
            return description["text"], described.captured_monotonic_ns if described else None
        if stage == "read_feedback":
            return reading.text if reading.status == "ready" else "No readable text found.", described.captured_monotonic_ns if described else None
        return None, None

    try:
        with (_open_proximity(settings) if proximity else nullcontext(None)) as monitor:
            with open_source(settings) as capture:
                processing_started = time.monotonic()
                logger.info("Offline preview active. P: local proximity; H: local help; S: stop speech; R: retry sensors; Q/Esc: quit." if offline else "Scene live active. D: one cloud snapshot; P: local proximity; S: stop speech; R: retry; Q/Esc: quit.")
                try:
                    while True:
                        tick = time.monotonic()
                        if settings.camera.source in ("picamera2", "imx500"):
                            packet = capture.read_packet()
                        else:
                            image = capture.read()
                            packet = None if image is None else FramePacket(count + 1, time.monotonic_ns(), image)
                        if packet is None:
                            raise CaptureError("The live camera stopped delivering frames. Check the connection.")
                        last_packet = packet
                        count += 1
                        if first_frame_ns is None:
                            first_frame_ns = packet.captured_monotonic_ns

                        if cloud_task is not None and cloud_task.done.is_set():
                            task, cloud_task = cloud_task, None
                            cooldown_until = time.monotonic() + max(5.0, settings.cloud.live_interval_s)
                            response, error = task.value if not task.failed and task.value is not None else (
                                None, None if task.cancel.is_set() else "Gemini description could not finish.")
                            described = cloud_packet
                            cancelled = response is None and error is None
                            if cloud_action == "read_text":
                                if cancelled:
                                    reading = replace(reading, status="cancelled", text="", error=None)
                                elif response is None or response.status not in ("ready", "no_text"):
                                    reading = replace(reading, status="failed", text="",
                                                      error="Gemini text reading is unavailable. Press R to retry.")
                                    cloud_paused = True
                                else:
                                    reading = response
                                record("read_text", reading.status, frame_id=described.frame_id)
                            elif cancelled:
                                description.update(status="cancelled", error=None,
                                                   elapsed_s=max(0.0, time.monotonic() - cloud_started))
                                record("describe", "cancelled", frame_id=described.frame_id)
                            elif response is None:
                                description.update(status="unavailable", error=error,
                                                   elapsed_s=max(0.0, time.monotonic() - cloud_started))
                                cloud_paused = True
                                record("describe", "unavailable", frame_id=described.frame_id,
                                       error=error, elapsed_s=description["elapsed_s"])
                            else:
                                description.update(status="ready", text=response.text, error=None,
                                                   model=response.model, elapsed_s=response.elapsed_s)
                                record("describe", "ready", frame_id=described.frame_id,
                                       captured_monotonic_ns=described.captured_monotonic_ns,
                                       text=response.text, model=response.model, elapsed_s=response.elapsed_s)
                                logger.info("Scene snapshot frame %d: %s", described.frame_id, response.text)
                            checkpoint()
                            if (cloud_action == "read_text" and reading.status in ("ready", "no_text")
                                    and cloud_generation == generation and local_task is None):
                                start_local("read_feedback")
                            elif (cloud_action == "describe" and not cancelled and (response is not None or proximity)
                                    and cloud_generation == generation and local_task is None):
                                start_local("scene_feedback")

                        if voice_task is not None and voice_task.done.is_set():
                            task, voice_task = voice_task, None
                            completed_stage = getattr(task, "kind", voice_stage)
                            if completed_stage == "transcribing":
                                cooldown_until = time.monotonic() + max(5.0, settings.cloud.live_interval_s)
                            if task.cancel.is_set() or voice_generation != generation:
                                voice_state.update(status="cancelled", error=None)
                                record("voice", "cancelled")
                            elif task.failed or task.value is None:
                                voice_paused = True
                                message = "Voice input is unavailable. Press R to retry; D and P remain available."
                                from vidi.audio.speech_to_text import SpeechRecognitionError
                                if isinstance(getattr(task, "failure", None), SpeechRecognitionError):
                                    message = str(task.failure)[:500] + " Press R to retry; D and P remain available."
                                voice_state.update(status="failed", error=message)
                                record("voice", "failed")
                            elif completed_stage == "recording":
                                voice_clip, task.value = task.value, None
                                voice_stage = "transcribe_ready"
                                voice_state.update(status="transcribing")
                                record("voice_record", "ready")
                            else:
                                response, task.value = task.value, None
                                voice_state.update(model=response.model, elapsed_s=response.elapsed_s)
                                if response.status == "ready":
                                    command = _parse_voice_command(response.text)
                                    if command.supported and command.action in ("describe", "read_text", "proximity", "stop", "help"):
                                        voice_pending = command.action
                                        voice_state.update(status="waiting_for_camera_request" if command.action in ("describe", "read_text") else "ready",
                                                           command=command.action, error=None)
                                        record("voice_command", "accepted", command=command.action)
                                        logger.info("Voice command accepted: %s.", command.action)
                                    else:
                                        voice_state.update(status="unknown_command", command=None,
                                                           error="Command not recognized. Press H for help.")
                                        record("voice_command", "unknown")
                                elif response.status in ("no_speech", "cancelled"):
                                    voice_state.update(status=response.status, error=None)
                                    record("voice", response.status)
                                else:
                                    voice_paused = True
                                    voice_state.update(status="failed", error="Voice recognition is unavailable. Press R to retry.")
                                    record("voice", "failed")
                            checkpoint()

                        if local_task is not None and local_task.done.is_set():
                            task, local_task = local_task, None
                            if task.cancel.is_set() or local_generation != generation:
                                if local_stage != "speech" or not receive_speech(task):
                                    speech_result = replace(speech_result, status="cancelled")
                                record(local_stage, speech_result.status if local_stage == "speech" else "cancelled")
                                checkpoint()
                            elif local_stage == "speech":
                                if task.failed:
                                    speech_result = SpeechResult(status="failed", error="Local speech could not finish.")
                                else:
                                    receive_speech(task)
                                if speech_result.text:
                                    logger.info("Requested feedback: %s", speech_result.text)
                                record("speech", speech_result.status, result=asdict(speech_result))
                                checkpoint()
                            else:
                                snapshot, sensor_text = task.value if not task.failed else (
                                    None, "Near-field sensors unavailable; left and right are unknown.")
                                if snapshot is not None:
                                    try:
                                        # Re-age the exact post-action pair; using latest() here
                                        # could resurrect a pre-action cache after a timeout.
                                        snapshot = _refresh_proximity(snapshot, settings)
                                        sensor_text = _format_proximity(snapshot, settings)
                                    except Exception:
                                        snapshot = None
                                        sensor_text = "Near-field sensors unavailable; left and right are unknown."
                                last_snapshot = snapshot
                                last_proximity_text = sensor_text
                                unavailable = local_stage == "scene_feedback" and description["status"] != "ready"
                                feedback_sensor_text = sensor_text + (" Scene description unavailable." if unavailable else "")
                                visual_text, captured_ns = feedback_image(local_stage)
                                text = compose_feedback(
                                    visual_text, feedback_sensor_text,
                                    max(0.0, (time.monotonic_ns() - captured_ns) / 1e9) if captured_ns is not None else 0.0,
                                    settings.speech.max_words,
                                )
                                record(local_stage, "ready", proximity=_snapshot_dict(snapshot),
                                       proximity_text=sensor_text)
                                if speak and text:
                                    totals["speech_attempts"] += 1
                                    speech_result = SpeechResult(status="pending", mode="playback", text=text)
                                    checkpoint()  # Persist intent before creating any audible process.
                                    local_stage = "speech"
                                    local_task = _Task(lambda cancel, snapshot=snapshot, sensor_text=sensor_text,
                                                      visual_text=visual_text, captured_ns=captured_ns,
                                                      unavailable=unavailable: speak_operation(
                                                          cancel, snapshot, sensor_text, visual_text,
                                                          captured_ns, unavailable))
                                else:
                                    if text:
                                        logger.info("Requested feedback: %s", text)
                                    checkpoint()

                        sensor_text = "Proximity not enabled."
                        if monitor is not None:
                            try:
                                sensor_text = _format_proximity(monitor.latest(), settings)
                            except Exception:
                                sensor_text = "Near-field sensors unavailable; left and right are unknown."
                        display_result = description if cloud_action == "describe" else {
                            "status": reading.status, "text": reading.text, "error": reading.error, "action": "read_text",
                        }
                        display = render_phase4_preview(
                            packet, cloud_packet, display_result, proximity_text=sensor_text,
                            cloud_busy=cloud_task is not None, audio_busy=local_task is not None,
                            paused=cloud_paused,
                            voice_status=voice_state["status"] if voice_commands else None,
                            voice_error=voice_state["error"] if voice_commands else None,
                            **({"offline": True} if offline else {}),
                        )
                        cv2.imshow(OFFLINE_WINDOW_TITLE if offline else VOICE_WINDOW_TITLE if voice_commands else WINDOW_TITLE, display)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            break
                        if offline and key in (ord("d"), ord("D"), ord("t"), ord("T"), ord("m"), ord("M")):
                            action = {"d": "describe", "t": "read_text", "m": "voice_record"}[chr(key).lower()]
                            totals["ignored_actions"] += 1
                            record(action, "disabled_offline")
                            checkpoint()
                        elif key in (ord("s"), ord("S")):
                            stop_narration()
                            record("stop", "completed")
                            checkpoint()
                        elif key in (ord("r"), ord("R")):
                            cloud_paused = False
                            voice_paused = False
                            if monitor is not None:
                                try:
                                    monitor.reconnect()
                                except Exception:
                                    logger.warning("Near-field sensors could not reconnect; channel statuses remain unknown.")
                            record("retry", "rearmed")
                            checkpoint()
                        elif key in (ord("d"), ord("D")):
                            if cloud_task is not None or local_task is not None or voice_busy() or cloud_paused or time.monotonic() < cooldown_until:
                                totals["ignored_actions"] += 1
                                record("describe", "ignored_busy_paused_or_cooldown")
                            else:
                                start_camera("describe", packet)
                        elif key in (ord("p"), ord("P")):
                            if voice_commands and voice_busy():
                                generation += 1
                                cancel_voice()
                            if not proximity or local_task is not None:
                                totals["ignored_actions"] += 1
                                record("proximity", "ignored_busy_or_disabled")
                            else:
                                generation += 1  # Supersede automatic narration from an outstanding D.
                                cancel_voice()
                                totals["local_reads"] += 1
                                start_local("proximity")
                        elif voice_commands and key in (ord("t"), ord("T")):
                            if cloud_task is not None or local_task is not None or voice_busy() or cloud_paused or time.monotonic() < cooldown_until:
                                totals["ignored_actions"] += 1
                                record("read_text", "ignored_busy_paused_or_cooldown")
                            else:
                                start_camera("read_text", packet)
                        elif (voice_commands or offline) and key in (ord("h"), ord("H")):
                            if local_task is None:
                                start_help()
                            else:
                                record("help", "ignored_busy")
                        elif voice_commands and key in (ord("m"), ord("M")):
                            if cloud_task is not None or local_task is not None or voice_busy() or voice_paused or time.monotonic() < cooldown_until:
                                totals["ignored_actions"] += 1
                                record("voice_record", "ignored_busy_paused_or_cooldown")
                            else:
                                generation += 1
                                voice_generation = generation
                                voice_stage = "recording"
                                voice_state.update(status="recording", command=None, error=None, model=None, elapsed_s=None)
                                record("voice_record", "pending")
                                checkpoint()
                                voice_task = _Task(lambda cancel: _record_voice(settings, cancel))
                                voice_task.kind = "recording"
                        advance_voice(packet)
                        time.sleep(max(0.0, 1.0 / settings.camera.fps - (time.monotonic() - tick)))
                except KeyboardInterrupt:
                    if last_packet is None:
                        raise
                    interrupted = True
                finally:
                    stop_narration()
                    if voice_task is not None:
                        record("voice", "cancelled")
                        voice_task.close()
                    if cloud_task is not None:
                        if cloud_action == "read_text":
                            reading = replace(reading, status="cancelled", text="", error=None)
                        else:
                            description["status"] = "cancelled"
                            description["elapsed_s"] = max(0.0, time.monotonic() - cloud_started)
                        record(cloud_action, "cancelled", frame_id=cloud_packet.frame_id)
                        cloud_task.close()
                    if local_task is not None:
                        local_task.close()
                        if local_stage == "speech" and local_task.done.is_set() and receive_speech(local_task):
                            record("speech", speech_result.status, result=asdict(speech_result))
                    processing_ended = time.monotonic()
    except cv2.error:
        raise SimulationError("The scene live preview failed. Check the desktop display and camera.") from None
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if last_packet is None:
        raise CaptureError("The live camera did not deliver a frame.")
    if settings.simulation.output_path is not None:
        export = cloud_packet or last_packet
        try:
            _save_image(settings.simulation.output_path, export.image_bgr)
            output = settings.simulation.output_path
        except SimulationError:
            if not interrupted:
                raise
            logger.error("The interrupted scene run could not save its final image.")
    try:
        checkpoint()
    except SimulationError:
        if not interrupted:
            raise
        logger.error("The interrupted scene run could not save its final report.")
    return SimulationResult(
        count, last_packet.image_bgr.shape[1], last_packet.image_bgr.shape[0], output,
        backend="none", startup_s=processing_started - started,
        processing_s=processing_ended - processing_started, report_path=settings.simulation.report_path,
        description=description["text"], description_status=description["status"],
        description_error=description["error"], description_model=description["model"],
        description_elapsed_s=description["elapsed_s"], speech=speech_result, interrupted=interrupted,
        ocr=reading,
    )
