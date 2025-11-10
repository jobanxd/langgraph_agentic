from fastapi import APIRouter, HTTPException
from models.chatbot_models import ChatbotAgentRequest, ChatbotAgentResponse
from services.chatbot_service import chatbot_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

@router.post("/chat", response_model=ChatbotAgentResponse)
async def chat(request: ChatbotAgentRequest):
    """
    Process chatbot message with session memory
    """
    try:
        response = chatbot_service.process_message(
            session_id=request.session_id,
            user_id=request.user_id,
            user_input=request.input_query
        )
        
        return ChatbotAgentResponse(response=response)
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session history
    """
    chatbot_service.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}