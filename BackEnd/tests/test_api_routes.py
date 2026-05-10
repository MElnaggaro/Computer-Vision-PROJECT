"""
API Routes Integration Tests
=============================
Tests all FastAPI endpoints for Speech, NLP, and Vision services.
"""

import base64
import sys
from pathlib import Path

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ── Speech Routes ──────────────────────────────────────────────────

def test_speech_root():
    response = client.get("/api/v1/speech/")
    assert response.status_code == 200
    assert response.json() == {"message": "Speech routes operational"}

def test_speech_transcribe_mocked(monkeypatch):
    """Test the POST /speech/transcribe endpoint with mocked speech."""
    # We mock SpeechRecognizer.listen_once to return a dummy SpeechResult
    from app.services.speech.speech_to_text import SpeechRecognizer, SpeechResult
    
    def mock_listen_once(self):
        return SpeechResult(text="what is tcp", language="en-US")
        
    monkeypatch.setattr(SpeechRecognizer, "listen_once", mock_listen_once)
    
    response = client.post("/api/v1/speech/transcribe")
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "what is tcp"
    assert data["language"] == "en-US"

# ── NLP Routes ─────────────────────────────────────────────────────

def test_nlp_root():
    response = client.get("/api/v1/nlp/")
    assert response.status_code == 200
    assert response.json() == {"message": "NLP routes operational"}

def test_nlp_classify():
    """Test the POST /nlp/classify endpoint."""
    payload = {"question": "How does the sliding window protocol work?"}
    response = client.post("/api/v1/nlp/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "How does the sliding window protocol work?"
    assert "topic" in data
    assert "topic_confidence" in data
    assert data["topic"] == "Computer Networks"

# ── Vision Routes ──────────────────────────────────────────────────

def test_vision_root():
    response = client.get("/api/v1/vision/")
    assert response.status_code == 200
    assert response.json() == {"message": "Vision routes operational"}

def _get_fake_base64_image():
    import numpy as np
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def test_vision_recognize_frame(monkeypatch):
    """Test the POST /vision/recognize-frame endpoint."""
    # We mock FaceDetector and FaceRecognizer to avoid heavy computation in API test
    from app.services.vision.face_detection import FaceDetector
    from app.services.vision.face_recognizer import FaceRecognizer
    
    monkeypatch.setattr(FaceDetector, "detect_faces", lambda *a, **k: [(10, 90, 90, 10)])
    monkeypatch.setattr(FaceRecognizer, "recognize_faces", lambda *a, **k: [{
        "name": "Unknown", "registered": False, "similarity": 0.2, "location": (10, 90, 90, 10)
    }])
    
    payload = {"image_base64": _get_fake_base64_image()}
    response = client.post("/api/v1/vision/recognize-frame", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["faces_detected"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Unknown"
