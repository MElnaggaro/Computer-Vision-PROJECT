"""
Tests — Speech-to-Text Module
===============================
Unit tests with mocked microphone (no real hardware needed).
Also includes a manual test entry point for live mic testing.

Run::

    cd BackEnd
    pytest tests/test_speech.py -v

Manual (needs mic)::

    python tests/test_speech.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.speech.speech_to_text import (
    SpeechAPIError,
    SpeechNotUnderstoodError,
    SpeechRecognizer,
    SpeechResult,
    SpeechTimeoutError,
    speech_to_text,
)

import speech_recognition as sr


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def recognizer() -> SpeechRecognizer:
    """Create a SpeechRecognizer with default settings."""
    return SpeechRecognizer(language="en-US", timeout=5, phrase_time_limit=10)


# ── SpeechRecognizer.listen_once tests ───────────────────────────────


class TestListenOnce:
    """Tests for ``SpeechRecognizer.listen_once()`` with mocked I/O."""

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "recognize_google", return_value="Can you explain convolution")
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_successful_recognition(
        self, mock_ambient, mock_listen, mock_google, mock_mic, recognizer
    ):
        """Happy path: speech is captured and transcribed."""
        result = recognizer.listen_once()

        assert isinstance(result, SpeechResult)
        assert result.text == "Can you explain convolution"
        assert result.language == "en-US"
        assert result.success is True

        # Verify Google API was called with explicit language
        mock_google.assert_called_once()
        call_kwargs = mock_google.call_args
        assert call_kwargs[1]["language"] == "en-US"

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "listen", side_effect=sr.WaitTimeoutError("timeout"))
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_timeout_raises_speech_timeout_error(
        self, mock_ambient, mock_listen, mock_mic, recognizer
    ):
        """When no speech is detected, SpeechTimeoutError is raised."""
        with pytest.raises(SpeechTimeoutError):
            recognizer.listen_once()

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "recognize_google", side_effect=sr.UnknownValueError())
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_unclear_audio_raises_not_understood(
        self, mock_ambient, mock_listen, mock_google, mock_mic, recognizer
    ):
        """When audio is unclear, SpeechNotUnderstoodError is raised."""
        with pytest.raises(SpeechNotUnderstoodError):
            recognizer.listen_once()

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "recognize_google", side_effect=sr.RequestError("API down"))
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_api_error_raises_speech_api_error(
        self, mock_ambient, mock_listen, mock_google, mock_mic, recognizer
    ):
        """When Google API fails, SpeechAPIError is raised."""
        with pytest.raises(SpeechAPIError):
            recognizer.listen_once()


# ── speech_to_text() convenience function tests ─────────────────────


class TestSpeechToTextFunction:
    """Tests for the backwards-compatible ``speech_to_text()`` function."""

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "recognize_google", return_value="Hello World")
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_returns_lowercase_text_on_success(
        self, mock_ambient, mock_listen, mock_google, mock_mic
    ):
        result = speech_to_text()
        assert result == "hello world"

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "listen", side_effect=sr.WaitTimeoutError("timeout"))
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_returns_none_on_timeout(self, mock_ambient, mock_listen, mock_mic):
        result = speech_to_text()
        assert result is None

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "recognize_google", side_effect=sr.UnknownValueError())
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_returns_none_on_unclear_audio(
        self, mock_ambient, mock_listen, mock_google, mock_mic
    ):
        result = speech_to_text()
        assert result is None


# ── SpeechRecognizer configuration tests ─────────────────────────────


class TestRecognizerConfig:
    """Verify default and custom configuration."""

    def test_default_language(self):
        rec = SpeechRecognizer()
        assert rec.language == "en-US"

    def test_default_timeout(self):
        rec = SpeechRecognizer()
        assert rec.timeout == 5

    def test_default_phrase_time_limit(self):
        rec = SpeechRecognizer()
        assert rec.phrase_time_limit == 10

    def test_custom_config(self):
        rec = SpeechRecognizer(language="en-GB", timeout=10, phrase_time_limit=20)
        assert rec.language == "en-GB"
        assert rec.timeout == 10
        assert rec.phrase_time_limit == 20


# ── SpeechResult dataclass tests ─────────────────────────────────────


class TestSpeechResult:
    def test_default_success_is_true(self):
        result = SpeechResult(text="hello", language="en-US")
        assert result.success is True

    def test_frozen_immutable(self):
        result = SpeechResult(text="hello", language="en-US")
        with pytest.raises(AttributeError):
            result.text = "modified"


# ── Manual test entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("  Manual Speech Test — Speak in English")
    print("=" * 50)

    recognizer = SpeechRecognizer(language="en-US", timeout=5, phrase_time_limit=10)

    try:
        result = recognizer.listen_once()
        print(f"\n✅ Text: {result.text}")
    except SpeechTimeoutError:
        print("\n❌ Timeout — no speech detected.")
    except SpeechNotUnderstoodError:
        print("\n❌ Could not understand audio.")
    except SpeechAPIError as exc:
        print(f"\n❌ API error: {exc}")
