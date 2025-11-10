from fastapi import APIRouter
from backend.models.chat_models import ChatRequest, ChatResponse
from backend.services.chat_service import process_query

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = process_query(request.query)
    return ChatResponse(response=response)
