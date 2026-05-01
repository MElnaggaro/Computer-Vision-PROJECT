from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def speech_root():
    return {"message": "Speech routes operational"}

@router.post("/stt")
async def speech_to_text():
    return {"status": "processing"}
