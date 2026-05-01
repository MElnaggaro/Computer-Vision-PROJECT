from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def vision_root():
    return {"message": "Vision routes operational"}

@router.post("/face-recognition")
async def face_recognition():
    return {"status": "processing"}

@router.post("/emotion-detection")
async def emotion_detection():
    return {"status": "processing"}

@router.post("/attendance")
async def attendance():
    return {"status": "processing"}
