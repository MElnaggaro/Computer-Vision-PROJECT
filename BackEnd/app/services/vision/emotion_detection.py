"""
Emotion Detection Service
=========================
Detects the dominant emotion using the `fer` library.
Model is loaded once at startup to ensure stable performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Classroom-friendly label map ─────────────────────────────────────
_LABEL_MAP: Dict[str, str] = {
    "happy":    "Happy",
    "sad":      "Sad",
    "neutral":  "Neutral",
    "angry":    "Angry",
    "fear":     "Anxious",
    "surprise": "Surprised",
    "disgust":  "Uncomfortable",
}

_MIN_FACE_PX = 40
EmotionResult = Dict[str, Any]

class EmotionDetector:
    """Predict the dominant emotion from a BGR face crop using FER.
    
    Initialized once to avoid repeated model loading and memory issues.
    """
    _instance: Optional['EmotionDetector'] = None
    _model: Optional[Any] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmotionDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, min_face_px: int = _MIN_FACE_PX) -> None:
        if not hasattr(self, 'initialized'):
            self.min_face_px = min_face_px
            try:
                # FER v25.x moved the class to fer.fer; older versions
                # re-exported it from the top-level package.
                try:
                    from fer.fer import FER
                except ImportError:
                    from fer import FER
                # mtcnn=False uses Haar Cascades for face detection (lightweight)
                # We already cropped the face, so we just want the emotion.
                self._model = FER(mtcnn=False)
                logger.info("FER Emotion model loaded successfully.")
            except ImportError as exc:
                logger.error("fer library is not installed: %s", exc)
                self._model = None
            self.initialized = True

    def predict(self, face_bgr: np.ndarray) -> EmotionResult:
        """Predict emotion from a BGR face crop.

        Strategy:
            1. Try FER ``detect_emotions`` on a padded version of the crop
               (padding helps FER's internal Haar cascade find the face).
            2. If Haar still misses, bypass FER's face detector and feed
               the crop directly to the underlying emotion CNN.
        """
        if face_bgr is None or face_bgr.size == 0:
            return self._fallback("Empty face crop")

        h, w = face_bgr.shape[:2]
        if h < self.min_face_px or w < self.min_face_px:
            return self._fallback(f"Face crop too small ({w}x{h} px)")

        if self._model is None:
            return self._fallback("FER model not loaded")

        try:
            # ── Attempt 1: FER with padded crop ──────────────────
            # FER expects BGR (it internally calls
            # cv2.cvtColor(img, COLOR_BGR2GRAY) for Haar detection).
            # Add padding so the Haar cascade has enough margin to
            # detect the face inside our tight crop.
            pad = int(max(h, w) * 0.3)
            padded = cv2.copyMakeBorder(
                face_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE,
            )
            emotions = self._model.detect_emotions(padded)

            if not emotions:
                # ── Attempt 2: bypass FER face detection ─────────
                # The face is already cropped; convert to the format
                # the emotion CNN expects (48×48 single-channel) and
                # run inference directly.
                emotions = self._classify_crop_directly(face_bgr)

            if not emotions:
                logger.warning(
                    "Emotion: no result from FER on %dx%d crop", w, h,
                )
                return self._fallback("No emotion detected in crop")

            # Take the first (best) face result
            emotion_data = emotions[0]["emotions"]

            # Find the dominant emotion
            dominant_raw = max(emotion_data, key=emotion_data.get)
            confidence = emotion_data[dominant_raw]

            raw_scores = {k: float(v) for k, v in emotion_data.items()}
            label = _LABEL_MAP.get(dominant_raw.lower(), dominant_raw.capitalize())

            return {
                "label": label,
                "confidence": round(float(confidence), 4),
                "raw_scores": raw_scores,
            }

        except Exception as exc:
            logger.warning("Emotion prediction failed: %s", exc)
            return self._fallback(str(exc))

    # ── Direct CNN classification (bypasses FER face detection) ───

    def _classify_crop_directly(self, face_bgr: np.ndarray):
        """Run the emotion CNN on a pre-cropped face without Haar detection.

        FER's ``detect_emotions`` first finds faces with a Haar cascade,
        which frequently fails on tight crops.  This method skips that
        step and feeds the crop straight into the emotion classifier.

        Returns:
            A list matching FER's ``detect_emotions`` format, or ``[]``.
        """
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

            # Use FER's own target size (typically 48×48)
            target_size = getattr(
                self._model, "_FER__emotion_target_size", (48, 48),
            )
            resized = cv2.resize(gray, target_size)
            normalized = resized.astype("float32") / 255.0

            # ── Path A: use FER's internal _classify_emotions ────
            # It expects shape (batch, h, w) and handles the rest.
            if hasattr(self._model, "_classify_emotions"):
                batch = np.expand_dims(normalized, 0)  # (1, 48, 48)
                preds = self._model._classify_emotions(batch)
                preds = np.array(preds)[0]
            # ── Path B: direct TFLite interpreter access ─────────
            elif hasattr(self._model, "_FER__tflite_interpreter"):
                interp = self._model._FER__tflite_interpreter
                inp_det = self._model._FER__tflite_input_details
                out_det = self._model._FER__tflite_output_details
                tensor = np.expand_dims(
                    np.expand_dims(normalized, 0), -1,
                ).astype(np.float32)  # (1, 48, 48, 1)
                interp.set_tensor(inp_det[0]["index"], tensor)
                interp.invoke()
                preds = interp.get_tensor(out_det[0]["index"])[0]
            else:
                return []

            # Map predictions to FER emotion labels (canonical order)
            emotion_labels = [
                "angry", "disgust", "fear", "happy",
                "sad", "surprise", "neutral",
            ]
            emotions_dict = {
                lbl: round(float(preds[i]), 2)
                for i, lbl in enumerate(emotion_labels)
            }
            h, w = face_bgr.shape[:2]
            return [{"box": np.array([0, 0, w, h]), "emotions": emotions_dict}]

        except Exception as exc:
            logger.warning("Direct CNN classification failed: %s", exc)
            return []

    @staticmethod
    def _fallback(reason: str = "") -> EmotionResult:
        if reason:
            logger.warning("Emotion fallback: %s", reason)
        return {
            "label": "Neutral",
            "confidence": 0.0,
            "raw_scores": {},
        }

    @staticmethod
    def normalize_label(raw_label: str) -> str:
        return _LABEL_MAP.get(raw_label.lower(), raw_label.capitalize())
