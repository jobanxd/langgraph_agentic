import logging
from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
from agents.graph import graph
from agents.state import AgentState
from utils.logging_utils import boxed_log

logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self):
        # In-memory session storage: {session_id: [messages]}
        self.sessions: Dict[str, list] = {}
    
    def process_message(self, session_id: str, user_id: str, user_input: str) -> str:
        """Process user message and return response"""
        
        # Get or create session history
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        logger.info(f"Processing message for session {session_id}, user {user_id}")
        
        # Add user message to history
        self.sessions[session_id].append(HumanMessage(content=user_input))
        logger.debug(f"User message added: {user_input}")

        # Create initial state
        initial_state: AgentState = {
            "messages": self.sessions[session_id],
            "session_id": session_id,
            "user_id": user_id,
            "next_agent": ""
        }
        logger.debug(f"Initial state prepared: {initial_state}")
        
        # Run graph
        result = graph.invoke(initial_state)
        
        # Extract response - get the last message
        messages = result.get("messages", [])
        if not messages:
            return "No response generated"
        
        ai_message = messages[-1]
        
        # Extract text content
        if hasattr(ai_message, 'content'):
            response_text = ai_message.content
        else:
            response_text = str(ai_message)
        
        # Update session - store all messages
        self.sessions[session_id] = list(messages)
        logger.debug(f"Session {session_id} updated with messages.")
        
        return response_text
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Singleton instance
chatbot_service = ChatbotService()