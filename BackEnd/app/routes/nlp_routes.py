from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def nlp_root():
    return {"message": "NLP routes operational"}

@router.post("/classify")
async def classify():
    return {"status": "processing"}
