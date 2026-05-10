"""
Attendance Service — Event-Based Logging
==========================================
Tracks student attendance and questions using an event-based log system.

Log format — each entry is a timestamped event::

    [
      {
        "event": "attendance",
        "student": "Mohammed_Ayman",
        "attendance": "Present",
        "emotion": "Happy",
        "emotion_confidence": 0.92,
        "timestamp": "2026-05-10T01:40:00+00:00",
        "registered": true
      },
      {
        "event": "question",
        "student": "Mohammed_Ayman",
        "question": "What is a semaphore?",
        "topic": "Operating System",
        "topic_confidence": 0.87,
        "timestamp": "2026-05-10T01:41:30+00:00"
      }
    ]

In-memory, each student also maintains a ``questions`` list for quick access::

    {
      "student": "Mohammed_Ayman",
      "attendance": "Present",
      "emotion": "Happy",
      "emotion_confidence": 0.92,
      "questions": [
        {"question": "What is a semaphore?", "topic": "Operating System", ...},
      ]
    }

Design decisions:
    • Uses an in-memory ``set`` for O(1) duplicate checks.
    • Event-based flat log for chronological traceability.
    • Per-student state dict with ``questions`` array for queries / overlay.
    • Unknown faces are logged with ``"attendance": "Not Registered"``.
    • Emotion and emotion_confidence are always included when available.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings

logger = logging.getLogger(__name__)

AttendanceRecord = Dict[str, Any]
EventRecord = Dict[str, Any]


class AttendanceService:
    """Mark attendance, record questions, and persist event-based JSON logs."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
    ) -> None:
        self.log_file = log_file or settings.ATTENDANCE_LOG_FILE
        self._marked: Set[str] = set()                      # student names marked so far
        self._events: List[EventRecord] = []                 # chronological event log
        self._student_state: Dict[str, AttendanceRecord] = {}  # per-student state
        self._has_unsaved_changes: bool = False

        # Load students already present in the log file to prevent duplicates across runs
        self._load_existing_students()

    def _load_existing_students(self) -> None:
        """Populate the marked set from the existing log file."""
        if not self.log_file.exists():
            return
        try:
            with open(self.log_file, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for record in data:
                            event_type = record.get("event", "attendance")
                            if event_type == "attendance":
                                name = record.get("student")
                                registered = record.get("registered", False)
                                if name and registered and name != "Unknown":
                                    self._marked.add(name)
            logger.info("Loaded %d previously marked students.", len(self._marked))
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not pre-load attendance log: %s", exc)

    # ── Public API ───────────────────────────────────────────────────

    def mark_attendance(
        self,
        name: str,
        registered: bool,
        similarity: float,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> Optional[AttendanceRecord]:
        """Record a student's attendance if not already marked.

        Creates both an event log entry and a per-student state record.

        Args:
            name:               Student name or ``"Unknown"``.
            registered:         Whether the student was recognised.
            similarity:         Recognition similarity (0–1).
            emotion:            Classroom-friendly emotion label (optional).
            emotion_confidence: Probability of the predicted emotion (0–1).

        Returns:
            The ``AttendanceRecord`` dict if newly marked, or ``None``
            if attendance was already recorded for this student.
        """
        # Skip duplicates for registered students
        if registered and self.already_marked(name):
            logger.debug("Attendance already marked for %s – skipping.", name)
            return None

        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Event log entry ──────────────────────────────────────────
        event: EventRecord = {
            "event": "attendance",
            "student": name,
            "attendance": "Present" if registered else "Not Registered",
            "timestamp": timestamp,
            "registered": registered,
        }

        if emotion is not None:
            event["emotion"] = emotion
        if emotion_confidence is not None:
            event["emotion_confidence"] = round(emotion_confidence, 4)

        self._events.append(event)

        # ── Per-student state ────────────────────────────────────────
        student_record: AttendanceRecord = {
            "student": name,
            "attendance": "Present" if registered else "Not Registered",
            "emotion": emotion or "Neutral",
            "emotion_confidence": round(emotion_confidence, 4) if emotion_confidence else 0.0,
            "timestamp": timestamp,
            "registered": registered,
            "questions": [],
        }
        self._student_state[name] = student_record

        if registered:
            self._marked.add(name)

        self._has_unsaved_changes = True
        logger.info(
            "Attendance marked: %s (%s, registered=%s, emotion=%s)",
            name,
            event["attendance"],
            registered,
            emotion or "N/A",
        )
        return student_record

    def add_question(
        self,
        student_name: str,
        question: str,
        topic: str,
        topic_confidence: float = 0.0,
    ) -> Optional[EventRecord]:
        """Record a question asked by a student.

        Creates a question event in the log and appends to the student's
        ``questions`` array.

        Args:
            student_name:    Name of the student who asked.
            question:        The transcribed question text.
            topic:           NLP-classified topic.
            topic_confidence: Classification confidence (0–1).

        Returns:
            The question event dict, or ``None`` if student is unknown.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Event log entry ──────────────────────────────────────────
        event: EventRecord = {
            "event": "question",
            "student": student_name,
            "question": question,
            "topic": topic,
            "topic_confidence": round(topic_confidence, 4),
            "timestamp": timestamp,
        }
        self._events.append(event)

        # ── Update per-student state ─────────────────────────────────
        if student_name in self._student_state:
            self._student_state[student_name]["questions"].append({
                "question": question,
                "topic": topic,
                "topic_confidence": round(topic_confidence, 4),
                "timestamp": timestamp,
            })

        self._has_unsaved_changes = True
        logger.info(
            "Question logged: %s asked '%s' → topic: %s (%.0f%%)",
            student_name,
            question[:50],
            topic,
            topic_confidence * 100,
        )
        print(f"\nSaved to logs.")
        return event

    def already_marked(self, name: str) -> bool:
        """Return ``True`` if the student was already marked present."""
        return name in self._marked

    def get_student_state(self, name: str) -> Optional[AttendanceRecord]:
        """Get the current state for a specific student.

        Returns:
            The student's record with attendance + questions, or ``None``.
        """
        return self._student_state.get(name)

    def get_active_student(self) -> Optional[str]:
        """Get the most recently marked registered student's name.

        Useful for push-to-talk: when a question is asked, attribute
        it to the most recently seen/active student.

        Returns:
            Student name string, or ``None`` if no registered students.
        """
        # Return the last registered student from events
        for event in reversed(self._events):
            if event.get("event") == "attendance" and event.get("registered"):
                return event["student"]
        return None

    def save_log(self) -> Path:
        """Persist all events to a JSON log file.

        Returns:
            The ``Path`` to the written log file.
        """
        if not self._has_unsaved_changes and self.log_file.exists():
            return self.log_file

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Merge with any existing log entries
        existing: List[EventRecord] = []
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                    if content:
                        existing = json.loads(content)
                        if not isinstance(existing, list):
                            existing = []
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Could not read existing log – overwriting: %s", exc)

        combined = existing + self._events

        with open(self.log_file, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, ensure_ascii=False)

        self._has_unsaved_changes = False
        logger.info("Saved %d events to %s", len(self._events), self.log_file)
        return self.log_file

    def get_student_summary(self) -> List[AttendanceRecord]:
        """Get a summary of all students with their questions.

        Returns a list of per-student records, each containing the
        attendance info and their ``questions`` array.

        Returns:
            List of student summary dicts.
        """
        return list(self._student_state.values())

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def records(self) -> List[EventRecord]:
        """Current session's event records (chronological)."""
        return list(self._events)

    @property
    def marked_students(self) -> Set[str]:
        """Set of student names that have been marked present."""
        return set(self._marked)

    def reset_session(self) -> None:
        """Clear in-memory state for a fresh attendance session."""
        self._marked.clear()
        self._events.clear()
        self._student_state.clear()
        self._has_unsaved_changes = False
        logger.info("Attendance session reset.")
