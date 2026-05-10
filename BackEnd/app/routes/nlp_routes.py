from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.orchestrator.question_pipeline import QuestionPipeline

router = APIRouter()

# Share the pipeline in-memory
_pipeline = QuestionPipeline()

class ClassifyRequest(BaseModel):
    question: str

class ClassifyResponse(BaseModel):
    question: str
    topic: str
    topic_confidence: float
    timestamp: str

@router.get("/")
async def nlp_root():
    return {"message": "NLP routes operational"}

@router.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest):
    """Classify a text question into one of the CS dataset topics."""
    try:
        result = _pipeline.process_text_question(payload.question)
        return ClassifyResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
