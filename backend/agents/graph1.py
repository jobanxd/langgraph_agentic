import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_core.tools import tool

from .state import AgentState
from .agentprofiles import AgentProfile
from .profiles.query_agent.tools import execute_query
from core.settings import settings
from utils.logging_utils import boxed_log

logger = logging.getLogger(__name__)


# AGENT PROFILES & LLM SETUP
root_profile = AgentProfile(agent_name="root_agent")
query_profile = AgentProfile(agent_name="query_agent")

root_llm = ChatGoogleGenerativeAI(
    model=root_profile.model_id,
    google_api_key=settings.GOOGLE_API_KEY
)

query_llm = ChatGoogleGenerativeAI(
    model=query_profile.model_id,
    google_api_key=settings.GOOGLE_API_KEY
)

# QUERY AGENT CREATION
query_agent = create_agent(
    query_llm,
    tools=[execute_query],
    system_prompt=f"{query_profile.instruction}\n\n{query_profile.description}"
)


# QUERY AGENT AS TOOL
@tool
def query_agent_tool(user_request: str) -> str:
    """
    Execute database queries and return results.
    Use this when you need to retrieve data from the database.
    
    Args:
        user_request: The user's query request that needs database access
        
    Returns:
        Query results as a formatted string
    """
    logger.info(f"QUERY AGENT TOOL CALLED with request: {user_request}")
    
    result = query_agent.invoke({
        "messages": [{"role": "user", "content": user_request}]
    })
    
    # Extract the final response
    final_message = result["messages"][-1]
    response_text = final_message.content[0]["text"]
    
    logger.info(f"QUERY AGENT TOOL RESPONSE: {response_text}")
    
    return response_text


# ROOT AGENT WITH QUERY AGENT AS TOOL CREATION
root_agent = create_agent(
    root_llm,
    tools=[query_agent_tool],
    system_prompt=f"{root_profile.instruction}\n\n{root_profile.description}"
)


# GRAPH NODES
def root_agent_node(state: AgentState) -> AgentState:
    logger.info("ROOT AGENT NODE ENTERED")

    result = root_agent.invoke(state)
    logger.info("Result from ROOT AGENT obtained: %s", result)
    
    final_message = result["messages"][-1]
    if isinstance(final_message.content, list):
        text = final_message.content[0].get("text", "")
        final_message.content = text
    boxed_log(f"ROOT AGENT FINAL RESPONSE: {final_message.content}", logger, level="info")
    
    return {
        "messages": [final_message],
        "next_agent": "END"
    }


# GRAPH CONSTRUCTION
def build_graph() -> StateGraph:
    """Build and compile the agent graph"""
    
    workflow = StateGraph(AgentState)
    
    # Single node - root agent handles everything
    workflow.add_node("root_agent", root_agent_node)
    
    # Simple linear flow: start -> root_agent -> end
    workflow.set_entry_point("root_agent")
    workflow.add_edge("root_agent", END)
    
    return workflow.compile()

graph = build_graph()