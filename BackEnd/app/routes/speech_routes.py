from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.speech.speech_to_text import SpeechRecognizer, SpeechError

router = APIRouter()

class TranscribeResponse(BaseModel):
    text: str
    language: str

@router.get("/")
async def speech_root():
    return {"message": "Speech routes operational"}

@router.post("/transcribe", response_model=TranscribeResponse)
async def speech_transcribe():
    """Activate the microphone, capture audio, and transcribe it to text."""
    recognizer = SpeechRecognizer()
    try:
        result = recognizer.listen_once()
        return TranscribeResponse(text=result.text, language=result.language)
    except SpeechError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

