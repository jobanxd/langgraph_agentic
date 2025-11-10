from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State that flows through the LangGraph.
    
    Attributes:
        messages: Conversation history (automatically merged by add_messages)
        session_id: Unique session identifier
        user_id: User identifier
        next_agent: Which agent should process next (used for routing)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    user_id: str
    next_agent: str  # Will be "query_agent", "END", etc.