"""Headless event controller: local wake input, requested cloud work, bounded audio.

Camera/audio own their hardware independently. Only the controller thread may
commit chess history. Generation checks and a shared lock make a late result
ineligible after wake, cancellation, mode changes, or privacy pause.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import logging
from pathlib import Path
import queue
import threading
import time
import uuid

from vidi.assistant.intent_router import Intent, route_intent
from vidi.core.frame_broker import FrameBroker, FrameUnavailable
from vidi.core.phase4 import _Task
from vidi.core.types import FramePacket

logger = logging.getLogger(__name__)


class AssistantError(RuntimeError):
    """A sanitized runtime or prerequisite failure."""


class AssistantController:
    def __init__(self, settings, *, mode="normal", cloud_enabled=False,
                 cloud_speech=False, speak=True, proximity=False, broker=None,
                 listener=None, speaker=None, transcriber=None, observer=None,
                 rectifier=None):
        self.settings = settings
        self.mode = mode
        self.cloud_enabled = cloud_enabled
        self.cloud_speech = cloud_speech
        self.speak_enabled = speak
        self.broker = broker or FrameBroker(settings)
        self.listener = listener
        self.speaker = speaker
        self.local_speaker = speaker
        self.transcriber = transcriber
        self.observer = observer
        self.rectifier = rectifier
        self.monitor = None
        self.proximity_enabled = proximity
        self.events = queue.Queue(maxsize=16)
        self.lock = threading.RLock()
        self.cloud_lock = threading.Lock()
        self.generation = 0
        self.job = self.speech_job = None
        self.retired = []
        self.job_generation = 0
        self.job_kind = None
        self.job_started_ns = 0
        self.cloud_paused = False
        self.privacy_paused = False
        self.quit = False
        self.session = self.automatic = None
        self.needs_chess_scan = True
        self.last_reply = ""
        self.speech_status = {"status": "not_requested"}
        self._camera_error_announced = False
        self.actions = deque(maxlen=120)
        self.totals = {"wake_events": 0, "cloud_requests": 0, "photos": 0, "dropped_events": 0}
        self.next_image_at = 0.
        self._started = time.monotonic()

    def record(self, action, status, *, elapsed_s=None):
        # Never log microphone clips, transcripts, key codes, image bytes, keys.
        item = {"action": action, "status": status, "monotonic_ns": time.monotonic_ns()}
        if elapsed_s is not None:
            item["elapsed_s"] = max(0., elapsed_s)
        self.actions.append(item)

    def post(self, action):
        if action in ("wake", "stop", "privacy_toggle", "quit"):
            # Called on input thread: stop audible playback before any cloud work.
            with self.lock:
                self.generation += 1
                if self.job:
                    self.job.cancel.set()
                if self.speech_job:
                    self.speech_job.cancel.set()
                if self.speaker:
                    self.speaker.stop()
                if self.local_speaker and self.local_speaker is not self.speaker:
                    self.local_speaker.stop()
        try:
            self.events.put_nowait(action)
        except queue.Full:
            self.totals["dropped_events"] += 1
            # Emergency privacy/quit is never lost behind ordinary requests.
            if action in ("stop", "privacy_toggle", "quit"):
                while True:
                    try:
                        self.events.get_nowait()
                    except queue.Empty:
                        break
                self.events.put_nowait(action)

    def start(self):
        if self.speaker is None:
            from vidi.audio.speech_factory import create_speech
            backend = self.settings.models.text_to_speech_backend
            self.speaker = create_speech(self.settings.speech, backend, cloud_enabled=self.cloud_speech)
            self.local_speaker = (create_speech(self.settings.speech) if backend == "gemini" else self.speaker)
        if self.proximity_enabled:
            from vidi.sensors.proximity import ProximityMonitor
            self.monitor = ProximityMonitor(self.settings.proximity).__enter__()
        self.broker.start()
        if self.mode == "chess":
            self._ensure_chess()
        if self.listener is not None:
            self.listener.start(lambda: self.post("wake"))
        self._say("vidi starting. Wait for the cue before your command. Space stops speech. F twelve pauses input.", local=True)

    def _ensure_chess(self):
        if self.session is None:
            from vidi.assistant.chess_automatic import AutomaticChessSession
            self.automatic = AutomaticChessSession(
                enabled=self.settings.chess.automatic,
                orientation=self.settings.chess.orientation,
                max_age_s=self.settings.chess.observation_max_age_s,
            )
            self.session = self.automatic.session
        if self.observer is None:
            from vidi.ai.chess_recognition import create_chess_observer
            self.observer = create_chess_observer(self.settings.cloud, self.settings.chess)
        if self.settings.chess.automatic and self.rectifier is None:
            from vidi.ai.board_geometry import BoardRectifier
            self.rectifier = BoardRectifier(self.settings.chess.calibration_path)
        if self.rectifier is not None and self.rectifier.orientation != self.settings.chess.orientation:
            raise AssistantError("Board calibration orientation does not match the chess profile.")

    def _cancel(self, *, discard_pending=True):
        with self.lock:
            self.generation += 1
            if self.job is not None:
                self.job.cancel.set()
                if not self.job.done.is_set():
                    self.retired.append(self.job)
                self.job = None
            if self.speech_job is not None:
                self.speech_job.cancel.set()
            if self.automatic is not None:
                self.automatic.cancel(discard_pending=discard_pending)
            if discard_pending and self.session is not None and self.session.pending is not None:
                self.session.reject()
            if self.speaker:
                self.speaker.stop()

    def _launch(self, kind, operation):
        self.retired = [task for task in self.retired if not task.done.is_set()]
        if self.job is not None or len(self.retired) >= 3:
            self._say("Still finishing the previous request. Use your wake phrase to interrupt.", local=True)
            return
        self.job_generation = self.generation
        self.job_kind = kind
        self.job_started_ns = time.monotonic_ns()
        self.job = _Task(operation)
        self.record(kind, "started")

    def _cloud(self, operation, cancel, *, image=False):
        while not cancel.is_set():
            if self.cloud_lock.acquire(timeout=.05):
                break
        else:
            return None
        try:
            if cancel.is_set() or not self.cloud_enabled or self.cloud_paused:
                return None
            if image:
                while time.monotonic() < self.next_image_at:
                    if cancel.wait(min(.05, self.next_image_at - time.monotonic())):
                        return None
            if cancel.is_set():
                return None
            self.totals["cloud_requests"] += 1
            try:
                return operation()
            finally:
                if image:
                    self.next_image_at = time.monotonic() + self.settings.cloud.live_interval_s
        finally:
            self.cloud_lock.release()

    def _say(self, text, *, local=False, prepare=None):
        if (self.mode == "chess" and self.settings.chess.backend == "fixture"
                and not text.startswith("SIMULATED.")):
            text = "SIMULATED. " + text
        self.last_reply = text
        if not self.speak_enabled:
            return
        from vidi.audio.speech_factory import render_chess_speech
        spoken = render_chess_speech(text) if self.mode == "chess" else text
        if self.speech_job:
            self.speech_job.cancel.set()
        speaker = self.local_speaker if local or not self.cloud_enabled else self.speaker
        if speaker is None:
            return
        generation = self.generation

        def operation(cancel):
            if cancel.is_set() or generation != self.generation:
                return None
            if speaker is self.speaker and self.settings.models.text_to_speech_backend == "gemini":
                return self._cloud(lambda: speaker.speak(spoken, cancel_event=cancel), cancel)
            deadline = time.monotonic() + 1.
            while not cancel.is_set():
                # A sensor snapshot can expire while waiting for prior playback.
                current = prepare() if prepare is not None else spoken
                result = speaker.speak(current, cancel_event=cancel)
                if (result.error != "Speech is busy; no utterance was queued."
                        or time.monotonic() >= deadline):
                    return result
                cancel.wait(.03)
            return None

        self.speech_job = _Task(operation)

    def _record_command(self, cancel):
        if self.listener is None:
            raise AssistantError("Wake input is not configured. Use the keyboard controls.")
        # An earcon, not spoken text, avoids narrating a command into the mic.
        self._cue(cancel)
        clip = self.listener.capture_next(self.settings.voice.record_seconds, cancel_event=cancel)
        if cancel.is_set():
            return None
        if self.transcriber is None:
            from vidi.audio.speech_to_text import GeminiCommandTranscriber
            self.transcriber = GeminiCommandTranscriber(self.settings.cloud, self.settings.voice)
        return self._cloud(lambda: self.transcriber.transcribe(clip, cancel_event=cancel), cancel)

    @staticmethod
    def _cue(cancel):
        import array
        import math
        from vidi.core.local_process import run_local_process
        data = array.array("h", (int(1200 * math.sin(i * 2 * math.pi * 880 / 16000)) for i in range(1280)))
        try:
            run_local_process(["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1"],
                              input_bytes=data.tobytes(), timeout_s=1., cancel_event=cancel)
        except Exception:
            # Cue availability is distinct from recording/recognition success.
            pass

    def _fresh(self, cancel, after_ns=None):
        return self.broker.wait_fresh(after_ns=after_ns or time.monotonic_ns(),
                                      cancel_event=cancel, max_age_s=self.settings.assistant.frame_max_age_s)

    def _visual(self, kind, cancel):
        # Wait for quota spacing BEFORE choosing the image, so waiting doesn't age it.
        def request():
            packet = self._fresh(cancel)
            if cancel.is_set():
                return None
            if kind == "read_text":
                from vidi.ai.gemini_text_recognition import GeminiTextRecognizer
                result = GeminiTextRecognizer(self.settings.cloud, self.settings.ocr).recognize(packet, cancel_event=cancel)
                return (packet, result)
            from vidi.ai.scene_description import GeminiSceneDescriber
            return packet, GeminiSceneDescriber(self.settings.cloud).describe(packet.image_bgr, cancel_event=cancel)
        return self._cloud(request, cancel, image=True)

    def _scan(self, cancel):
        if self.settings.chess.backend == "fixture":
            packet = self._fresh(cancel)
            return self.observer.observe(packet, cancel_event=cancel)
        def request():
            first = self._fresh(cancel)
            if not self.settings.chess.automatic:
                return self.observer.observe(first, cancel_event=cancel)
            first = replace(first, image_bgr=self.rectifier.rectify(first.image_bgr))
            if cancel.wait(.3):
                return None
            second = self._fresh(cancel, after_ns=first.captured_monotonic_ns + 300_000_000)
            second = replace(second, image_bgr=self.rectifier.rectify(second.image_bgr))
            # Abstain on motion/occlusion; agreement of model results alone is insufficient.
            import cv2
            import numpy as np
            a = cv2.cvtColor(cv2.resize(first.image_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(cv2.resize(second.image_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
            if np.mean(cv2.absdiff(a, b)) > 6 or np.mean(cv2.absdiff(a, b) > 25) > .04:
                raise AssistantError("Board view changed. Hold still, remove hands, and request a new scan.")
            return self.observer.observe_pair(first, second, cancel_event=cancel)
        return self._cloud(request, cancel, image=True)

    def handle(self, intent):
        if isinstance(intent, str):
            intent = Intent(intent)
        action = intent.action
        self.record(action, "requested")
        if action == "quit":
            self._cancel()
            self.quit = True
            return
        if action in ("privacy_toggle", "privacy_pause"):
            pause = not self.privacy_paused if action == "privacy_toggle" else True
            self._cancel()
            if pause:
                self.privacy_paused = True
                self.broker.pause()
                if self.listener:
                    self.listener.pause()
            else:
                try:
                    if self.listener:
                        self.listener.resume()
                    self.broker.resume()
                    self.privacy_paused = False
                except Exception:
                    self.privacy_paused = True
                    self.broker.pause()
                    if self.listener:
                        self.listener.pause()
                    self._say("Input is still unavailable. Check the headset, then press F twelve to retry.", local=True)
                    return
            self._say("Camera and microphone paused. Press F twelve to resume." if pause else "Camera and microphone resumed.", local=True)
            return
        if action == "stop":
            self._cancel()
            return
        if self.privacy_paused:
            return
        if action == "wake":
            self.totals["wake_events"] += 1
            self._cancel(discard_pending=False)
            if not self.cloud_enabled or self.cloud_paused:
                self._say("Cloud commands unavailable. Use the local recovery keys, or retry.", local=True)
            else:
                self._launch("transcription", self._record_command)
            return
        if action.startswith("mode_"):
            self._cancel()
            if action == "mode_chess":
                try:
                    self._ensure_chess()
                except Exception:
                    self._say("Chess is unavailable. Check the local rules package and board calibration.", local=True)
                    return
            self.mode = action[5:]
            self.needs_chess_scan = True
            self._say(f"{self.mode.capitalize()} mode. " + ("Request a fresh board scan before board queries." if self.mode == "chess" else "Ready."), local=True)
            return
        if action == "retry":
            self._cancel()
            self.cloud_paused = False
            self.broker.pause()
            self.broker.resume()
            if self.monitor:
                self.monitor.reconnect()
            self._say("Requests rearmed. Please ask again.", local=True)
            return
        if action == "help":
            self._say("Use your wake phrase before each request. Describe, read text, take a photo, proximity, chess mode, normal mode, or stop. In chess, ask what moved, whose turn, or what is on e four.", local=True)
            return
        if action == "repeat":
            self._say(self.last_reply or "Nothing to repeat.")
            return
        if action == "forget":
            self.last_reply = ""
            self._say("Recent reply cleared. Confirmed chess history is preserved.", local=True)
            return
        if action == "proximity":
            def proximity(cancel):
                if self.monitor is None:
                    return "Proximity sensors are not enabled."
                from vidi.sensors.proximity import format_proximity, refresh_snapshot
                snapshot = self.monitor.fresh(time.monotonic_ns(), cancel_event=cancel)
                return format_proximity(refresh_snapshot(snapshot, self.settings.proximity.stale_after_s), self.settings.proximity.nearby_mm)
            self._launch("proximity", proximity)
            return
        if action == "capture":
            def capture(cancel):
                from vidi.core.controller import _save_image
                packet = self._fresh(cancel)
                if cancel.is_set():
                    return None
                path = self.settings.assistant.photo_dir_path / ("vidi-" + uuid.uuid4().hex + ".png")
                _save_image(path, packet.image_bgr)
                self.totals["photos"] += 1
                return "Photo saved locally. It was not uploaded."
            self._launch("capture", capture)
            return
        if self.mode == "chess":
            self._ensure_chess()
            if action == "scan":
                if self.settings.chess.backend != "fixture" and not self._can_cloud():
                    return
                self._scan_generation = self.automatic.generation
                self._launch("scan", self._scan)
            elif action == "confirm":
                if self.session.pending is None:
                    self._say("No proposal is waiting for confirmation.", local=True)
                else:
                    result = self.session.confirm(self.session.pending.proposal_id)
                    self.needs_chess_scan = result.status != "confirmed"
                    self._say(result.message)
            elif action == "reject":
                self._cancel()
                self._say("Proposal rejected. Confirmed history is unchanged.", local=True)
            elif action in ("turn", "last_move", "board", "piece"):
                self._say("Request a fresh board scan before resuming chess information." if self.needs_chess_scan else self.session.query(action, intent.square))
            else:
                self._say("Command not recognized. Say help for the available commands.", local=True)
        elif action in ("describe", "read_text"):
            if self._can_cloud():
                self._launch(action, lambda cancel: self._visual(action, cancel))
        else:
            self._say("Command not recognized. Say help for the available commands.", local=True)

    def _can_cloud(self):
        if not self.cloud_enabled or self.cloud_paused:
            self._say("Cloud requests unavailable. Local photos, proximity and controls still work.", local=True)
            return False
        return True

    def tick(self):
        while True:
            try:
                action = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle(action)
        with self.lock:
            if self.speech_job is not None and self.speech_job.done.is_set():
                speech = self.speech_job
                self.speech_job = None
                if speech.value is not None and not speech.failed:
                    result = speech.value
                    self.speech_status = {"status": result.status, "backend": result.backend,
                                          "voice": result.voice, "elapsed_s": result.elapsed_s}
                    self.record("speech", result.status)
                    if result.status == "failed":
                        logger.warning("Speech unavailable. Check the configured engine and audio output.")
                        if result.backend == "gemini":
                            self.cloud_paused = True
            if self.broker.error and not self._camera_error_announced:
                self._camera_error_announced = True
                self._say("Camera unavailable. Local controls remain active. Check the connection, then press R to retry.", local=True)
            elif not self.broker.error:
                self._camera_error_announced = False
            task = self.job
            if task is not None and task.done.is_set():
                self.job = None
                kind = self.job_kind
                if task.cancel.is_set() or self.job_generation != self.generation:
                    self.record(kind, "cancelled")
                    return
                if task.failed:
                    self.record(kind, "failed")
                    if kind in ("describe", "read_text", "scan", "transcription"):
                        self.cloud_paused = True
                    message = (str(task.failure) if isinstance(task.failure, (AssistantError, FrameUnavailable))
                               else "Request unavailable. Check the device or connection, then retry.")
                    self._say(message, local=True)
                    return
                value = task.value
                self.record(kind, "completed", elapsed_s=(time.monotonic_ns() - self.job_started_ns) / 1e9)
                if value is None:
                    return
                if kind == "transcription":
                    if value.status == "ready":
                        # Transcription is transient and never copied into the report.
                        self.handle(route_intent(value.text, self.mode))
                    else:
                        if value.status == "failed":
                            self.cloud_paused = True
                        self._say(value.error or "No command was recognized. Please ask again.", local=True)
                elif kind == "scan":
                    if isinstance(value, tuple):
                        if any(item.status == "failed" for item in value):
                            self.cloud_paused = True
                            self._say("Chess observation unavailable. Check the connection and retry.", local=True)
                            return
                        transition = self.automatic.apply_pair(*value, generation=self._scan_generation,
                                                               cancel_event=task.cancel)
                    else:
                        transition = self.session.propose(value)
                        if value.status == "failed":
                            self.cloud_paused = True
                    self.needs_chess_scan = transition.status not in ("confirmed", "no_change")
                    self._say(transition.message)
                elif kind in ("describe", "read_text"):
                    packet, result = value
                    if getattr(result, "status", "ready") == "failed":
                        self.cloud_paused = True
                        self._say(result.error or "Reading unavailable. Retry when connected.", local=True)
                    else:
                        age = max(0., (time.monotonic_ns() - packet.captured_monotonic_ns) / 1e9)
                        self._say(f"Camera snapshot from {age:.1f} seconds ago: " + (result.text or "No readable text found."))
                elif kind == "proximity" and self.monitor is not None:
                    from vidi.sensors.proximity import format_proximity
                    def fresh_feedback():
                        return format_proximity(self.monitor.latest(), self.settings.proximity.nearby_mm)
                    self._say(fresh_feedback(), local=True, prepare=fresh_feedback)
                else:
                    self._say(value, local=True)

    def snapshot(self):
        from vidi.core.controller import _observed_rate
        return {"schema_version": 6, "workflow": "assistant", "mode": self.mode,
                "simulated": self.settings.camera.source == "sample" or (self.mode == "chess" and self.settings.chess.backend == "fixture"),
                "privacy_paused": self.privacy_paused, "cloud_enabled": self.cloud_enabled,
                "cloud_paused": self.cloud_paused, "totals": dict(self.totals),
                "actions": list(self.actions), "frames_processed": self.broker.frames,
                "capture_fps": _observed_rate(self.broker.frames, self.broker.first_ns, self.broker.last_ns),
                "processing_s": max(0., time.monotonic() - self._started),
                "chess": self.session.snapshot() if self.session else None,
                "speech": dict(self.speech_status),
                "acceptance": "Physical wake, audio, chess and cold-boot acceptance remain pending."}

    def close(self):
        self._cancel()
        resources = [self.listener, self.broker, self.monitor, self.speech_job,
                     *self.retired, self.speaker]
        if self.local_speaker is not self.speaker:
            resources.append(self.local_speaker)
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    logger.warning("A local assistant resource could not finish cleanup.")


def run_assistant(settings, *, mode="normal", cloud_enabled=False,
                  cloud_speech=False, wake=True, speak=True, proximity=False,
                  run_seconds=None):
    """Run with an optional desktop preview, never requiring stdin or waitKey."""
    from vidi.core.controller import _save_report
    keyboard = listener = None
    previous_term = None
    next_keyboard_retry = 0.
    if wake:
        from vidi.audio.continuous_input import AudioInput, ContinuousListener
        from vidi.audio.wake_word import WakeDetector
        assets = settings.assistant
        if assets.wake_model_dir_path is None or assets.wake_keywords_path is None:
            raise AssistantError("Configure approved wake model/keyword paths first, or explicitly use --no-wake with recovery keys.")
        listener = ContinuousListener(AudioInput(settings.voice.device),
                                      WakeDetector(assets.wake_model_dir_path, assets.wake_keywords_path,
                                                   threshold=assets.wake_threshold))
    controller = AssistantController(settings, mode=mode, cloud_enabled=cloud_enabled,
                                     cloud_speech=cloud_speech, speak=speak,
                                     proximity=proximity, listener=listener)
    try:
        if threading.current_thread() is threading.main_thread():
            import signal
            previous_term = signal.signal(signal.SIGTERM, lambda *_: controller.post("quit"))
        if settings.assistant.keyboard_device_path is not None:
            from vidi.core.keyboard_control import KeyboardControl
            keyboard = KeyboardControl(settings.assistant.keyboard_device_path, controller.post).start()
        elif not settings.simulation.preview:
            raise AssistantError("Headless operation requires a selected USB recovery keyboard. Configure assistant.keyboard_device_path.")
        controller.start()
        logger.info("Assistant active in %s mode. Wake input is %s; cloud requests are %s.", mode,
                    "local" if wake else "disabled", "requested only" if cloud_enabled else "disabled")
        deadline = time.monotonic() + run_seconds if run_seconds is not None else None
        while not controller.quit and (deadline is None or time.monotonic() < deadline):
            controller.tick()
            if keyboard is not None and keyboard.error and not controller.privacy_paused:
                controller.handle("privacy_pause")
            if keyboard is not None and keyboard.error and time.monotonic() >= next_keyboard_retry:
                # Reopen only the explicitly selected identity; never resume mic/camera automatically.
                next_keyboard_retry = time.monotonic() + 2.
                try:
                    keyboard.close()
                    from vidi.core.keyboard_control import KeyboardControl
                    keyboard = KeyboardControl(settings.assistant.keyboard_device_path, controller.post).start()
                except (OSError, ValueError):
                    pass
            if listener is not None and listener.error and not controller.privacy_paused:
                controller.handle("privacy_pause")
            if settings.simulation.preview:
                import cv2
                try:
                    frame = controller.broker.latest().image_bgr
                    cv2.imshow("vidi assistant: D scene / T read / B scan / M command / Space stop / F12 pause / Q quit", frame)
                except FrameUnavailable:
                    pass
                key = cv2.waitKeyEx(1)
                controls = {ord("d"): "describe", ord("t"): "read_text", ord("b"): "scan",
                            ord("m"): "wake", ord("p"): "proximity", ord("c"): "confirm",
                            ord("x"): "reject", ord("h"): "help", ord("r"): "retry",
                            ord("1"): "mode_normal", ord("2"): "mode_chess", ord(" "): "stop",
                            ord("q"): "quit", 27: "quit", 65481: "privacy_toggle", 0x7B0000: "privacy_toggle"}
                if key in controls:
                    controller.post(controls[key])
            time.sleep(.01)
    except KeyboardInterrupt:
        controller.post("quit")
    except AssistantError:
        raise
    except Exception:
        raise AssistantError("Assistant startup or device access failed. Check the local assets, input permissions and selected devices.") from None
    finally:
        if previous_term is not None:
            import signal
            signal.signal(signal.SIGTERM, previous_term)
        if keyboard:
            try:
                keyboard.close()
            except OSError:
                logger.warning("Recovery keyboard cleanup could not complete.")
        controller.close()
        if settings.simulation.preview:
            import cv2
            cv2.destroyAllWindows()
        if settings.simulation.report_path:
            _save_report(settings.simulation.report_path, controller.snapshot())
    return controller.snapshot()
