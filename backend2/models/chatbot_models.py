from pydantic import BaseModel, Field

class ChatbotAgentRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier", example="abc123")
    user_id: str = Field(..., description="Unique user identifier", example="user_001")
    input_query: str = Field(..., description="User's input text for the chatbot", example="Hello, how are you?")

class ChatbotAgentResponse(BaseModel):
    response: str = Field(..., description="Chatbot's generated reply", example="I'm doing great! How can I help you today?")
