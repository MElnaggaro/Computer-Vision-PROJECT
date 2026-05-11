"""
Live Full Pipeline Integration Test
=====================================
A REAL end-to-end manual integration test that opens the actual webcam
and exercises the **complete** Smart Classroom vision pipeline:

    Real webcam frame
    → real face detection   (FaceDetector)
    → real face recognition (FaceRecognizer + EncodingManager)
    → real face tracking    (FaceTracker)
    → real emotion detection(EmotionDetector via EmotionTracker)
    → real attendance mark  (AttendanceService → classroom_log.json)

Usage (from BackEnd/):
    python -m tests.live_full_pipeline_test

Controls:
    Q / ESC  → Quit
    R        → Reset attendance session
    B        → Rebuild face encodings from scratch
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure BackEnd root is on sys.path ──────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import cv2
import numpy as np

# ── Project service imports ─────────────────────────────────────────
from app.core.config import settings
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer
from app.services.vision.face_tracker import FaceTracker
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.emotion_detection import EmotionDetector
from app.services.vision.emotion_tracker import EmotionTracker
from app.services.vision.attendance_service import AttendanceService

# ── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_pipeline_test")


# ═══════════════════════════════════════════════════════════════════
#  Overlay / Drawing Helpers
# ═══════════════════════════════════════════════════════════════════

# Color palette (BGR)
_CLR_GREEN      = (0, 220, 100)
_CLR_RED        = (60, 60, 230)
_CLR_YELLOW     = (0, 220, 255)
_CLR_CYAN       = (230, 200, 0)
_CLR_WHITE      = (255, 255, 255)
_CLR_DARK_BG    = (30, 30, 30)
_CLR_OVERLAY_BG = (0, 0, 0)

_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_SM  = 0.50
_FONT_SCALE_MD  = 0.60
_FONT_THICKNESS = 1


def _draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: tuple,
    color: tuple = _CLR_WHITE,
    bg_color: tuple = _CLR_OVERLAY_BG,
    font_scale: float = _FONT_SCALE_SM,
    thickness: int = _FONT_THICKNESS,
    alpha: float = 0.6,
) -> None:
    """Draw text with a semi-transparent background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = org
    pad = 4
    # Background rect
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        bg_color,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Text
    cv2.putText(frame, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_face_overlay(
    frame: np.ndarray,
    result: Dict[str, Any],
    emotion_label: str,
    emotion_conf: float,
    attendance_status: str,
) -> None:
    """Draw bounding box + info labels for one detected face."""
    top, right, bottom, left = result["location"]
    name = result.get("name", "Unknown")
    similarity = result.get("similarity", 0.0)
    registered = result.get("registered", False)

    # Box colour
    if registered:
        box_clr = _CLR_GREEN
    elif name == "Unknown":
        box_clr = _CLR_RED
    else:
        box_clr = _CLR_YELLOW

    # Bounding box with rounded corners effect (thick + thin)
    cv2.rectangle(frame, (left, top), (right, bottom), box_clr, 2)
    # Corner accents
    corner_len = 15
    for (cx, cy), (dx, dy) in [
        ((left, top), (1, 1)),
        ((right, top), (-1, 1)),
        ((left, bottom), (1, -1)),
        ((right, bottom), (-1, -1)),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), box_clr, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), box_clr, 3)

    # ── Labels below the box ─────────────────────────────────────
    y_offset = bottom + 18
    line_gap = 20

    display_name = name.replace("_", " ") if name != "Unknown" else "Unknown"
    _draw_text_with_bg(
        frame, f"{display_name}", (left, y_offset),
        color=_CLR_WHITE, font_scale=_FONT_SCALE_MD,
    )
    y_offset += line_gap

    _draw_text_with_bg(
        frame, f"Confidence: {similarity * 100:.1f}%", (left, y_offset),
        color=_CLR_CYAN,
    )
    y_offset += line_gap

    emotion_display = f"Emotion: {emotion_label} ({emotion_conf * 100:.0f}%)"
    _draw_text_with_bg(
        frame, emotion_display, (left, y_offset),
        color=_CLR_YELLOW,
    )
    y_offset += line_gap

    _draw_text_with_bg(
        frame, f"Status: {attendance_status}", (left, y_offset),
        color=_CLR_GREEN if attendance_status == "Present" else _CLR_RED,
    )


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    num_faces: int,
    marked_count: int,
    total_students: int,
    session_events: int,
) -> None:
    """Draw the heads-up-display panel in the top-left corner."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {num_faces}",
        f"Marked: {marked_count}/{total_students}",
        f"Events: {session_events}",
        "──────────────────────────",
        "Q/ESC: Quit | R: Reset | B: Rebuild",
    ]
    y = 25
    for line in lines:
        _draw_text_with_bg(frame, line, (10, y), color=_CLR_WHITE, font_scale=_FONT_SCALE_SM)
        y += 20


# ═══════════════════════════════════════════════════════════════════
#  Main Pipeline Runner
# ═══════════════════════════════════════════════════════════════════

class LivePipelineTest:
    """Runs the full live pipeline with real services."""

    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.fps_buffer: deque = deque(maxlen=30)

        # ── Instantiate real services ────────────────────────────
        logger.info("Initialising services...")

        # 1. Encoding manager
        try:
            self.encoding_manager = EncodingManager()
            result = self.encoding_manager.ensure_fresh()
            logger.info(
                "EncodingManager ready — %d students, status: %s",
                len(self.encoding_manager.names), result["status"],
            )
        except Exception as exc:
            logger.error("Failed to initialise EncodingManager: %s", exc)
            raise SystemExit(1)

        # 2. Face detection
        self.face_detector = FaceDetector()
        logger.info("FaceDetector ready (model=%s)", self.face_detector.model)

        # 3. Face recognition
        self.face_recognizer = FaceRecognizer(
            encoding_manager=self.encoding_manager,
        )
        logger.info("FaceRecognizer ready (tolerance=%.2f)", self.face_recognizer.tolerance)

        # 4. Face tracker
        self.face_tracker = FaceTracker()
        logger.info("FaceTracker ready")

        # 5. Emotion detector + tracker
        try:
            self.emotion_detector = EmotionDetector()
            if self.emotion_detector._model is None:
                logger.warning(
                    "⚠ FER model did not load — emotion will fall back to 'Neutral'."
                )
            else:
                logger.info("EmotionDetector ready (FER model loaded)")
        except Exception as exc:
            logger.warning("EmotionDetector init failed: %s — falling back", exc)
            self.emotion_detector = None

        self.emotion_tracker = EmotionTracker(
            emotion_interval=settings.EMOTION_DETECTION_INTERVAL,
            buffer_size=settings.EMOTION_BUFFER_SIZE,
            max_stale_frames=settings.EMOTION_MAX_STALE_FRAMES,
            detector=self.emotion_detector,
            min_stable_samples=settings.EMOTION_MIN_STABLE_SAMPLES,
        )
        logger.info("EmotionTracker ready")

        # 6. Attendance service (writes to classroom_log.json)
        self.attendance_service = AttendanceService()
        self.attendance_service.reset_session()
        logger.info(
            "AttendanceService ready — log file: %s",
            self.attendance_service.log_file,
        )

        # Per-track attendance status cache
        self._attendance_cache: Dict[int, str] = {}

    # ── Webcam ───────────────────────────────────────────────────

    def _open_camera(self) -> bool:
        """Open the default webcam and verify it works."""
        logger.info("Opening webcam (index 0)...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback without CAP_DSHOW
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            logger.error("❌ Cannot open webcam — is it connected and not in use?")
            return False

        # Set resolution for comfortable viewing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("✅ Webcam opened — resolution %dx%d", w, h)
        return True

    def _release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ── Actions ──────────────────────────────────────────────────

    def _reset_session(self) -> None:
        """Reset the attendance session (keyboard R)."""
        self.attendance_service.reset_session()
        self.face_tracker.reset()
        self.emotion_tracker.reset()
        self._attendance_cache.clear()
        logger.info("🔄 Session reset — all attendance cleared.")

    def _rebuild_encodings(self) -> None:
        """Rebuild all face encodings from disk (keyboard B)."""
        logger.info("🔨 Rebuilding face encodings...")
        result = self.encoding_manager.rebuild_all_encodings()
        logger.info(
            "✅ Rebuild complete — %d students, status: %s",
            len(self.encoding_manager.names), result["status"],
        )

    # ── Pipeline per-frame ───────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the full pipeline on a single frame and return the annotated frame."""
        self.frame_count += 1

        # Resize for faster processing (half resolution)
        h, w = frame.shape[:2]
        scale = 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # 1. Face detection
        face_locations = self.face_detector.detect_faces(small)

        # Scale locations back to original frame coords
        scaled_locations = [
            (int(t / scale), int(r / scale), int(b / scale), int(l / scale))
            for (t, r, b, l) in face_locations
        ]

        # 2. Face recognition
        if scaled_locations:
            raw_results = self.face_recognizer.recognize_faces(
                frame, scaled_locations,
            )
        else:
            raw_results = []

        # 3. Face tracking (temporal stabilization)
        stable_results = self.face_tracker.update(raw_results)

        # 4. Per-face: emotion detection + attendance
        for result in stable_results:
            track_id = result.get("track_id", -1)
            top, right, bottom, left = result["location"]

            # Clamp to frame bounds
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)

            face_crop = frame[top:bottom, left:right]

            # ── Emotion ──────────────────────────────────────────
            if face_crop.size > 0:
                emotion_result = self.emotion_tracker.update(
                    track_id=track_id,
                    face_crop=face_crop,
                    frame_count=self.frame_count,
                )
            else:
                emotion_result = {"label": "Neutral", "confidence": 0.0}

            emotion_label = emotion_result.get("label", "Neutral")
            emotion_conf = emotion_result.get("confidence", 0.0)

            # ── Attendance ───────────────────────────────────────
            name = result.get("name", "Unknown")
            registered = result.get("registered", False)
            similarity = result.get("similarity", 0.0)
            attendance_ready = result.get("attendance_ready", False)

            # Determine attendance status for overlay
            if registered and self.attendance_service.already_marked(name):
                att_status = "Present"
            elif registered and attendance_ready:
                # Mark attendance NOW — first-time recognition confirmed
                record = self.attendance_service.mark_attendance(
                    name=name,
                    registered=True,
                    similarity=similarity,
                    emotion=emotion_label,
                    emotion_confidence=emotion_conf,
                )
                if record is not None:
                    att_status = "Present"
                    logger.info(
                        "✅ ATTENDANCE MARKED: %s (%.1f%% confidence, emotion=%s)",
                        name, similarity * 100, emotion_label,
                    )
                else:
                    att_status = "Present"
            elif not registered:
                att_status = "Not Registered"
            else:
                att_status = "Recognizing..."

            # Also log stable emotion once enough samples accumulate
            if registered and self.emotion_tracker.is_stable(track_id):
                smoothed = self.emotion_tracker.get_smoothed(track_id)
                if smoothed:
                    self.attendance_service.log_emotion(
                        name=name,
                        mood=smoothed["label"],
                        samples=self.emotion_tracker.sample_count(track_id),
                    )

            self._attendance_cache[track_id] = att_status

            # ── Draw overlay ─────────────────────────────────────
            _draw_face_overlay(
                frame, result,
                emotion_label, emotion_conf,
                att_status,
            )

        # 5. HUD
        fps = self._compute_fps()
        _draw_hud(
            frame,
            fps=fps,
            num_faces=len(stable_results),
            marked_count=len(self.attendance_service.marked_students),
            total_students=len(self.encoding_manager.names),
            session_events=len(self.attendance_service.records),
        )

        return frame

    # ── FPS ──────────────────────────────────────────────────────

    def _compute_fps(self) -> float:
        now = time.perf_counter()
        self.fps_buffer.append(now)
        if len(self.fps_buffer) < 2:
            return 0.0
        elapsed = self.fps_buffer[-1] - self.fps_buffer[0]
        return (len(self.fps_buffer) - 1) / elapsed if elapsed > 0 else 0.0

    # ── Main loop ────────────────────────────────────────────────

    def run(self) -> None:
        """Open the webcam and run the live pipeline until the user quits."""
        if not self._open_camera():
            return

        window_name = "Live Pipeline Test  |  Smart Classroom"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        print("\n" + "=" * 60)
        print("  LIVE FULL PIPELINE TEST — Smart Classroom Assistant")
        print("=" * 60)
        print(f"  Students loaded : {len(self.encoding_manager.names)}")
        print(f"  Known names     : {', '.join(self.encoding_manager.names) or '(none)'}")
        print(f"  Log file        : {self.attendance_service.log_file}")
        print(f"  Tolerance       : {self.face_recognizer.tolerance:.2f}")
        print(f"  Detection model : {self.face_detector.model}")
        print()
        print("  Controls:")
        print("    Q / ESC  →  Quit")
        print("    R        →  Reset attendance session")
        print("    B        →  Rebuild face encodings")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame — retrying...")
                    time.sleep(0.1)
                    continue

                annotated = self._process_frame(frame)
                cv2.imshow(window_name, annotated)

                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):      # Q or ESC
                    logger.info("Quit requested by user.")
                    break
                elif key in (ord("r"), ord("R")):
                    self._reset_session()
                elif key in (ord("b"), ord("B")):
                    self._rebuild_encodings()

        except KeyboardInterrupt:
            logger.info("Interrupted by Ctrl+C.")
        finally:
            self._release_camera()
            cv2.destroyAllWindows()

            # ── Print final summary ──────────────────────────────
            print("\n" + "=" * 60)
            print("  SESSION SUMMARY")
            print("=" * 60)
            summary = self.attendance_service.get_student_summary()
            if summary:
                for s in summary:
                    name = s.get("student", "?")
                    att = s.get("attendance", "?")
                    emo = s.get("emotion", "?")
                    print(f"  • {name:<25} {att:<18} Emotion: {emo}")
            else:
                print("  (No students were marked)")
            print(f"\n  Total events  : {len(self.attendance_service.records)}")
            print(f"  Total frames  : {self.frame_count}")
            print(f"  Log file      : {self.attendance_service.log_file}")
            print("=" * 60)

            # Verify the log file was written
            log_path = self.attendance_service.log_file
            if log_path.exists():
                try:
                    data = json.loads(log_path.read_text(encoding="utf-8"))
                    print(f"\n  ✅ Log file verified: {len(data)} event(s) persisted.")
                except Exception as exc:
                    print(f"\n  ⚠ Log file exists but could not parse: {exc}")
            else:
                print(f"\n  ⚠ Log file not found at {log_path}")


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎓 Smart Classroom — Live Full Pipeline Integration Test\n")

    # Pre-flight checks
    print("Pre-flight checks:")

    # Check OpenCV
    print(f"  [✓] OpenCV {cv2.__version__}")

    # Check face_recognition
    try:
        import face_recognition
        print(f"  [✓] face_recognition available")
    except ImportError:
        print("  [✗] face_recognition NOT installed — aborting.")
        sys.exit(1)

    # Check FER
    try:
        try:
            from fer.fer import FER
        except ImportError:
            from fer import FER
        print(f"  [✓] FER (emotion detection) available")
    except ImportError:
        print("  [!] FER not installed — emotion detection will fall back to Neutral")

    # Check student data
    if settings.STUDENTS_FACES_DIR.exists():
        students = [d.name for d in settings.STUDENTS_FACES_DIR.iterdir() if d.is_dir()]
        print(f"  [✓] Student database: {len(students)} student(s)")
        for s in students:
            print(f"      └─ {s}")
    else:
        print(f"  [!] No student database found at {settings.STUDENTS_FACES_DIR}")

    # Check encoding cache
    if settings.ENCODINGS_DIR.exists():
        pkls = list(settings.ENCODINGS_DIR.glob("*.pkl"))
        print(f"  [✓] Encoding cache: {len(pkls)} file(s)")
    else:
        print(f"  [!] No encoding cache at {settings.ENCODINGS_DIR}")

    print()

    # Run
    test = LivePipelineTest()
    test.run()
