"""Requested chess observations and explicit confirmation in a responsive UI.

Workers acquire observations only. The UI thread alone proposes and confirms
game changes; cancelled or superseded work cannot mutate the session. Fixture
mode never captures audio or constructs a cloud client.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import logging
import textwrap
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vidi.audio.text_to_speech import LocalSpeech
from vidi.camera.capture import CaptureError
from vidi.core.chess_types import BoardObservation
from vidi.core.phase4 import _Task
from vidi.core.types import FramePacket, SpeechResult

if TYPE_CHECKING:
    from vidi.core.configuration import AppSettings

logger = logging.getLogger(__name__)
ACTION_LIMIT = 120
WINDOW_TITLE = "vidi chess: B scan | C confirm | X reject | L last | W turn | A board | H help | S stop | R retry | Q quit"
HELP_TEXT = (
    "B scans the board. A scan only proposes a change. After checking the physical board, "
    "your partner presses C to confirm, or X to reject. L reads the last confirmed move. "
    "W reads whose turn. A summarizes the confirmed board. S cancels. R retries. Q quits."
)


def _create_session(settings):
    from vidi.assistant.chess_session import ChessSession, ChessSessionError
    from vidi.core.controller import SimulationError

    try:
        return ChessSession(orientation=settings.chess.orientation,
                            max_age_s=settings.chess.observation_max_age_s)
    except ChessSessionError as error:
        raise SimulationError(str(error)) from None


def _create_observer(settings):
    from vidi.ai.chess_recognition import create_chess_observer
    return create_chess_observer(settings.cloud, settings.chess)


def _record_command(settings, cancel):
    from vidi.audio.speech_to_text import MicrophoneRecorder
    return MicrophoneRecorder(settings.voice).record(cancel_event=cancel)


def _transcribe_command(settings, clip, cancel):
    from vidi.audio.speech_to_text import GeminiCommandTranscriber
    return GeminiCommandTranscriber(settings.cloud, settings.voice).transcribe(clip, cancel_event=cancel)


def _parse_command(text):
    from vidi.assistant.chess_commands import parse_chess_command
    return parse_chess_command(text)


def render_chess_preview(packet, observed, state, message, *, status, voice_status,
                         paused, simulated, voice_commands, scan_pending=False):
    """Current pixels, exact scanned pixels, and confirmed state stay distinct."""
    def bounded_copy(image):
        scale = min(1., 640 / image.shape[1], 480 / image.shape[0])
        if scale < 1.:
            return cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
        return image.copy()

    live = bounded_copy(packet.image_bgr)
    analysis = np.zeros_like(live) if observed is None else bounded_copy(observed.image_bgr)
    width, height = max(live.shape[1], analysis.shape[1]), max(live.shape[0], analysis.shape[0])

    def padded(image):
        canvas = np.zeros((height, width, 3), dtype=image.dtype)
        x, y = (width - image.shape[1]) // 2, (height - image.shape[0]) // 2
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
        return canvas

    live, analysis = padded(live), padded(analysis)

    def draw(image, texts):
        wrapped = [line for value in texts for line in textwrap.wrap(str(value), max(20, image.shape[1] // 9))]
        height = min(image.shape[0], len(wrapped) * 20 + 8)
        cv2.rectangle(image, (0, 0), (image.shape[1], height), (25, 25, 25), -1)
        for index, text in enumerate(wrapped):
            y = 18 + index * 20
            if y >= image.shape[0]:
                break
            cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                        min(.5, max(.2, image.shape[1] / 1200)), (245, 245, 245), 1, cv2.LINE_AA)

    prefix = "SIMULATED FIXTURE. " if simulated else ""
    draw(live, [prefix + f"LIVE CAMERA frame {packet.frame_id}",
                "B scan / C confirm / X reject / L last / W turn / A board",
                ("M command / " if voice_commands else "") + "H help / S cancel / R retry / Q quit",
                "Voice: " + voice_status if voice_commands else "Voice not enabled."])
    age = max(0., (time.monotonic_ns() - observed.captured_monotonic_ns) / 1e9) if observed else 0.
    pending = state.get("pending")
    confirmed = (f"Confirmed moves: {len(state.get('confirmed_moves', []))}. Last: {state.get('last_move') or 'No moves yet.'}"
                 if state.get("initialized") else "Not initialized: scan the standard starting position.")
    draw(analysis, [prefix + (f"{'SUBMITTED' if scan_pending else 'SCANNED'} frame {observed.frame_id}, {age:.1f}s old" if observed else "No board scanned."),
                    "Requests paused; press R." if paused else "Status: " + status,
                    "CONFIRMED: " + str(confirmed),
                    "Unconfirmed proposal: C after verification / X reject." if pending else "No pending proposal.",
                    message])
    return cv2.hconcat((live, analysis))


def run_chess_live(settings: AppSettings, *, speak: bool, voice_commands: bool, open_source):
    """B/M are explicit requests; a different C/M-confirm action commits state."""
    from vidi.core.controller import SimulationError, SimulationResult, _observed_rate, _save_image, _save_report

    if not settings.simulation.preview:
        raise SimulationError("Chess live mode requires the desktop preview.")
    if settings.models.object_detection_backend not in (None, "none"):
        raise SimulationError("Chess live mode requires the object detector to be off.")
    simulated = settings.chess.backend == "fixture"
    if simulated and voice_commands:
        raise SimulationError("Simulated chess fixtures cannot enable microphone or cloud voice commands.")
    session = _create_session(settings)
    observer = None
    actions = deque(maxlen=ACTION_LIMIT)
    totals = {"actions": 0, "scan_requests": 0, "cloud_requests": 0,
              "audio_requests": 0, "speech_attempts": 0, "ignored_actions": 0}
    cloud_task = record_task = speech_task = None
    cloud_kind = None
    cloud_frame = last_packet = observed_frame = last_observation = None
    cloud_generation = record_generation = generation = 0
    voice_clip = voice_pending = None
    voice_status = "not_requested"
    paused = interrupted = False
    cooldown_until = 0.
    last_transition = None
    last_message = "SIMULATED. " if simulated else ""
    last_message += "Show the standard starting position and press B. No game state is confirmed yet."
    last_status = "not_requested"
    speech_result = SpeechResult()
    count = ready = 0
    first_ns = None
    output = None
    started = processing_started = time.monotonic()
    processing_ended = None

    def label(text):
        return ("SIMULATED. " if simulated else "") + text

    def record(action, status, **fields):
        totals["actions"] += 1
        actions.append({"action": action, "status": status, "monotonic_ns": time.monotonic_ns(), **fields})

    def checkpoint():
        if settings.simulation.report_path is not None:
            _save_report(settings.simulation.report_path, {
                "schema_version": 5, "mode": "chess_live", "continuous": True,
                "source": settings.camera.source, "simulated": simulated,
                "frames_processed": count, "last_preview_frame_id": last_packet.frame_id if last_packet else None,
                "capture_fps": _observed_rate(count, first_ns, last_packet.captured_monotonic_ns if last_packet else None),
                "startup_s": processing_started - started,
                "processing_s": (processing_ended if processing_ended is not None else time.monotonic()) - processing_started,
                "session": session.snapshot(), "transition": last_transition,
                "observation": asdict(last_observation) if last_observation is not None else None,
                "message": last_message, "status": last_status,
                "speech": asdict(speech_result), "voice_commands_enabled": voice_commands,
                "voice_status": voice_status, "requests_paused": paused,
                "cloud_min_interval_s": max(5., settings.cloud.live_interval_s),
                "actions": list(actions), "action_totals": dict(totals), "action_records_limit": ACTION_LIMIT,
                "action_records_truncated": totals["actions"] > len(actions),
                "action_records_dropped": totals["actions"] - len(actions),
                "output_path": str(output) if output else None,
                "output_frame_id": (cloud_frame or observed_frame or last_packet).frame_id if output else None,
                "interrupted": interrupted,
            })

    def feedback(text, action, status="ready", *, audible=True):
        nonlocal last_message, last_status, speech_task, speech_result
        last_message, last_status = label(text), status
        record(action, status)
        logger.info("Requested chess feedback: %s", last_message)
        if audible and speak and speech_task is None:
            totals["speech_attempts"] += 1
            speech_result = SpeechResult(status="pending", mode="playback", text=last_message)
            checkpoint()
            text = last_message

            def operation(cancel):
                if cancel.is_set():
                    return SpeechResult(status="cancelled", mode="playback", text=text)
                return LocalSpeech(settings.speech, settings.models.text_to_speech_backend).speak(text, cancel_event=cancel)

            speech_task = _Task(operation)
        checkpoint()

    def transition_feedback(transition, action):
        nonlocal last_transition
        last_transition = asdict(transition)
        text = transition.message
        if transition.status == "proposed":
            if simulated and text.startswith("Simulated observation. "):
                text = text.removeprefix("Simulated observation. ")
            if text.startswith("Possible move: "):
                text = text.removeprefix("Possible move: ")
            if text == "Standard starting position detected. Awaiting confirmation before tracking moves.":
                text = "Standard starting position."
            text = "Proposed: " + text.rstrip(". ") + ". Partner: verify, then C confirms or X rejects."
        last_transition["message"] = label(text)
        feedback(text, action, transition.status)

    def cancel_work(*, announce=False):
        nonlocal generation, voice_clip, voice_pending, voice_status, last_status, last_message
        generation += 1
        active = cloud_task is not None or record_task is not None or voice_clip is not None or voice_pending is not None
        for task in (cloud_task, record_task, speech_task):
            if task is not None:
                task.cancel.set()
        voice_clip = voice_pending = None
        if active:
            voice_status = "cancelled"
        rejected = session.reject()
        if announce:
            last_status, last_message = "cancelled", label("Cancelled. Any unconfirmed proposal was discarded.")
            record("stop", "cancelled")
            checkpoint()
        return rejected

    def scan_operation(cancel, packet):
        nonlocal observer
        if cancel.is_set():
            return None
        if observer is None:
            observer = _create_observer(settings)
        if cancel.is_set():
            return None
        return observer.observe(packet, cancel_event=cancel)

    def busy():
        return cloud_task is not None or record_task is not None or voice_clip is not None or voice_pending is not None

    def start_scan(packet):
        nonlocal cloud_task, cloud_kind, cloud_generation, cloud_frame, generation, last_status, last_message
        if session.pending is not None:
            pending_scan_feedback()
            return
        generation += 1
        cloud_generation = generation
        cloud_kind = "scan"
        cloud_frame = FramePacket(packet.frame_id, packet.captured_monotonic_ns, packet.image_bgr.copy())
        totals["scan_requests"] += 1
        totals["cloud_requests"] += not simulated
        last_status, last_message = "pending", label("Board observation pending. Confirmed position is unchanged.")
        record("scan", "pending", frame_id=packet.frame_id)
        checkpoint()
        cloud_task = _Task(lambda cancel, packet=cloud_frame: scan_operation(cancel, packet))

    def pending_scan_feedback():
        totals["ignored_actions"] += 1
        feedback("Confirm C or reject X first. The current proposal is unchanged.", "scan", "ignored_pending")

    def local_action(action, square=None):
        nonlocal last_transition
        if action == "confirm":
            if speech_task is not None:
                totals["ignored_actions"] += 1
                feedback("Speech is active. Wait for it to finish, then press C to confirm.", action, "ignored_busy", audible=False)
                return
            pending = session.pending
            if pending is None:
                feedback("There is no unconfirmed proposal.", action, "no_pending")
            else:
                transition_feedback(session.confirm(pending.proposal_id, now_ns=time.monotonic_ns()), action)
        elif action == "reject":
            transition_feedback(cancel_work(), action)
        elif action == "stop":
            cancel_work(announce=True)
        elif action == "help":
            feedback(HELP_TEXT + (" Press M to record one chess command." if voice_commands else ""), action)
        elif action == "repeat":
            text = last_message.removeprefix("SIMULATED. ") if simulated else last_message
            feedback(text, action)
        else:
            text = session.query(action, square)
            if session.initialized and action in ("piece", "turn"):
                text = "From the last confirmed position: " + text
            if session.pending is not None:
                text += " A board observation still awaits confirmation."
            feedback(text, action)

    try:
        with open_source(settings) as capture:
            processing_started = time.monotonic()
            try:
                while True:
                    tick = time.monotonic()
                    if settings.camera.source in ("picamera2", "imx500"):
                        packet = capture.read_packet()
                    else:
                        image = capture.read()
                        packet = None if image is None else FramePacket(count + 1, time.monotonic_ns(), image)
                    if packet is None:
                        raise CaptureError("The chess camera stopped delivering frames.")
                    last_packet = packet
                    count += 1
                    first_ns = packet.captured_monotonic_ns if first_ns is None else first_ns

                    if speech_task is not None and speech_task.done.is_set():
                        task, speech_task = speech_task, None
                        speech_result = task.value if not task.failed and task.value is not None else SpeechResult(status="failed", error="Local chess feedback could not finish.")
                        record("speech", speech_result.status)
                        checkpoint()

                    if record_task is not None and record_task.done.is_set():
                        task, record_task = record_task, None
                        if task.cancel.is_set() or record_generation != generation:
                            voice_status = "cancelled"
                        elif task.failed or task.value is None:
                            voice_status, paused = "failed", True
                            feedback("Command input is unavailable. Press R to retry. Keyboard queries remain available.", "voice", "failed", audible=False)
                        else:
                            voice_clip, task.value = task.value, None
                            voice_status = "transcribing"
                        checkpoint()

                    if cloud_task is not None and cloud_task.done.is_set():
                        task, cloud_task = cloud_task, None
                        if not simulated:
                            cooldown_until = time.monotonic() + max(5., settings.cloud.live_interval_s)
                        if task.cancel.is_set() or cloud_generation != generation:
                            voice_status = "cancelled" if cloud_kind == "audio" else voice_status
                            record(cloud_kind, "cancelled")
                        elif cloud_kind == "scan":
                            observed_frame = cloud_frame
                            if (task.failed or not isinstance(task.value, BoardObservation)
                                    or task.value.frame_id != cloud_frame.frame_id
                                    or task.value.captured_monotonic_ns != cloud_frame.captured_monotonic_ns
                                    or task.value.simulated != simulated):
                                last_observation = BoardObservation(cloud_frame.frame_id, cloud_frame.captured_monotonic_ns,
                                                                    status="failed", error="Chess observation is unavailable.", simulated=simulated)
                            else:
                                last_observation = task.value
                            if last_observation.status == "failed":
                                paused = True
                                feedback("Board recognition is unavailable. Press R to retry. Confirmed position is unchanged.", "scan", "unavailable")
                            else:
                                ready += last_observation.status == "ready"
                                transition_feedback(session.propose(last_observation, now_ns=time.monotonic_ns()), "scan")
                        elif task.failed or task.value is None:
                            voice_status, paused = "failed", True
                            feedback("Voice recognition is unavailable. Press R to retry.", "voice", "failed", audible=False)
                        else:
                            response, task.value = task.value, None
                            if response.status == "ready":
                                command = _parse_command(response.text)
                                if command.action in ("scan", "confirm", "reject", "turn", "last_move", "board", "repeat", "piece", "stop", "help"):
                                    voice_pending = command
                                    voice_status = "waiting_for_camera_request" if command.action == "scan" else "ready"
                                    record("voice_command", "accepted", command=command.action, square=command.square)
                                    logger.info("Chess voice command accepted: %s.", command.action)
                                else:
                                    voice_status = "unknown_command"
                                    feedback("Command not recognized. Press H for chess help.", "voice", "unknown_command", audible=False)
                            elif response.status in ("no_speech", "cancelled"):
                                voice_status = response.status
                                record("voice", response.status)
                            else:
                                voice_status, paused = "failed", True
                                feedback("Voice recognition is unavailable. Press R to retry.", "voice", "failed", audible=False)
                        checkpoint()

                    scan_pending = cloud_task is not None and cloud_kind == "scan"
                    cv2.imshow(WINDOW_TITLE + (" | M command" if voice_commands else ""), render_chess_preview(
                        packet, cloud_frame if scan_pending else observed_frame, session.snapshot(), last_message,
                        status="pending" if scan_pending else last_status, voice_status=voice_status,
                        paused=paused, simulated=simulated, voice_commands=voice_commands, scan_pending=scan_pending))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("s"), ord("S")):
                        local_action("stop")
                    elif key in (ord("x"), ord("X")):
                        local_action("reject")
                    elif key in (ord("r"), ord("R")):
                        paused = False
                        record("retry", "rearmed")
                        checkpoint()
                    elif key in (ord("b"), ord("B")):
                        if session.pending is not None:
                            pending_scan_feedback()
                        elif busy() or speech_task is not None or paused or time.monotonic() < cooldown_until:
                            totals["ignored_actions"] += 1
                            record("scan", "ignored_busy_paused_or_cooldown")
                        else:
                            start_scan(packet)
                    elif key in (ord("m"), ord("M")) and voice_commands:
                        if busy() or speech_task is not None or paused or time.monotonic() < cooldown_until:
                            totals["ignored_actions"] += 1
                            record("voice_record", "ignored_busy_paused_or_cooldown")
                        else:
                            generation += 1
                            record_generation = generation
                            voice_status = "recording"
                            record("voice_record", "pending")
                            checkpoint()
                            record_task = _Task(lambda cancel: _record_command(settings, cancel))
                    else:
                        action = {ord("c"): "confirm", ord("l"): "last_move", ord("w"): "turn",
                                  ord("a"): "board", ord("h"): "help"}.get(key + 32 if 65 <= key <= 90 else key)
                        if action is not None:
                            local_action(action)

                    # Stages are launched after S/Q/X for this frame were handled.
                    if voice_clip is not None and record_generation == generation:
                        clip, voice_clip = voice_clip, None
                        cloud_kind, cloud_generation = "audio", generation
                        voice_status = "transcribing"
                        totals["cloud_requests"] += 1
                        totals["audio_requests"] += 1
                        record("voice_transcribe", "pending")
                        checkpoint()
                        cloud_task = _Task(lambda cancel, clip=clip: _transcribe_command(settings, clip, cancel))
                    elif voice_pending is not None and cloud_generation == generation:
                        command = voice_pending
                        if command.action != "scan":
                            voice_pending = None
                            local_action(command.action, command.square)
                        elif session.pending is not None:
                            voice_pending = None
                            voice_status = "ready"
                            pending_scan_feedback()
                        elif not paused and cloud_task is None and speech_task is None and time.monotonic() >= cooldown_until:
                            voice_pending = None
                            voice_status = "ready"
                            start_scan(packet)
                    time.sleep(max(0., 1. / settings.camera.fps - (time.monotonic() - tick)))
            except KeyboardInterrupt:
                if last_packet is None:
                    raise
                interrupted = True
            finally:
                active_intent = cloud_task is not None or record_task is not None or voice_clip is not None or voice_pending is not None
                pending_proposal = session.pending is not None
                if cloud_task is not None:
                    record(cloud_kind, "cancelled", **({"frame_id": cloud_frame.frame_id} if cloud_kind == "scan" else {}))
                if record_task is not None:
                    record("voice_record", "cancelled")
                if voice_clip is not None or voice_pending is not None:
                    record("voice_command", "cancelled")
                cancel_work()
                for task in (record_task, cloud_task, speech_task):
                    if task is not None:
                        task.close()
                if speech_task is not None and speech_task.done.is_set() and not speech_task.failed and speech_task.value is not None:
                    speech_result = speech_task.value
                    record("speech", speech_result.status)
                elif speech_task is not None:
                    speech_result = SpeechResult(status="cancelled", mode="playback", text=speech_result.text)
                    record("speech", "cancelled")
                if active_intent or pending_proposal:
                    last_status = "cancelled"
                    last_message = label("Stopped. Pending requests and any unconfirmed proposal were discarded.")
                record("quit", "completed")
                processing_ended = time.monotonic()
    except cv2.error:
        raise SimulationError("The chess preview failed. Check the desktop display and camera.") from None
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if last_packet is None:
        raise CaptureError("The chess camera did not deliver a frame.")
    if settings.simulation.output_path is not None:
        _save_image(settings.simulation.output_path, (cloud_frame or observed_frame or last_packet).image_bgr)
        output = settings.simulation.output_path
    checkpoint()
    return SimulationResult(count, last_packet.image_bgr.shape[1], last_packet.image_bgr.shape[0], output,
                            backend="none", simulated=simulated, ready_frames=ready,
                            startup_s=processing_started - started, processing_s=processing_ended - processing_started,
                            report_path=settings.simulation.report_path, description=last_message,
                            description_status=last_status, speech=speech_result, interrupted=interrupted)
