import logging
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from .state import AgentState
from .agentprofiles import AgentProfile
from .profiles.query_agent.tools import execute_query
from core.settings import settings
from utils.logging_utils import boxed_log

logger = logging.getLogger(__name__)

# Load agent profiles
root_profile = AgentProfile(agent_name="root_agent")
query_profile = AgentProfile(agent_name="query_agent")

# Initialize LLMs
root_llm = ChatGoogleGenerativeAI(
    model=root_profile.model_id,
    google_api_key=settings.GOOGLE_API_KEY,)
query_llm = ChatGoogleGenerativeAI(
    model=query_profile.model_id,
    google_api_key=settings.GOOGLE_API_KEY,)

# Create query agent with tools
query_agent = create_agent(
    query_llm,
    tools=[execute_query],
)

def root_agent_node(state: AgentState) -> AgentState:
    """Root agent decides routing or processes response"""
    messages = state["messages"]

    logger.info("ROOT AGENT NODE ENTERED")
    logger.info(f"Current next_agent state: {state.get('next_agent')}")
    
    # Check if we're processing query_agent response
    if state.get("next_agent") == "process_query_result":
        logger.info("ROOT AGENT: Processing query_agent results")

        query_result = messages[-1].content[0]["text"]
        logger.info(f"QUERY AGENT RESULT TEXT: {query_result}")
        # Root agent interprets query results
        system_prompt = f"{root_profile.instruction}\n\n{root_profile.description}"
        post_procesing_prompt = f"""Based on the query results below, generate a natural, helpful response to answer the user's question.

Query Results: 
{query_result}

Provide a clear, conversational answer with insights from the data."""
        final_response = root_llm.invoke([{"role": "system", "content": system_prompt},
                                          {"role": "user", "content": post_procesing_prompt}])
        boxed_log(f"ROOT AGENT FINAL RESPONSE: {final_response.content}", logger, level="info")
        return {"messages": [final_response], "next_agent": "END"}
    
    # Initial routing decision
    logger.info("ROOT AGENT: Making routing decision")
    system_prompt = f"{root_profile.instruction}\n\n{root_profile.description}"
    routing_prompt = f"""Decide if you need query_agent for database access.
Respond ONLY: "query_agent" or "answer_directly"

User: {messages[-1].content}"""

    response = root_llm.invoke([{"role": "system", "content": system_prompt}, 
                                 {"role": "user", "content": routing_prompt}])
    
    decision = response.content.strip().lower()
    logger.info(f"ROOT AGENT DECISION: {decision}")
    
    if "query_agent" in decision:
        logger.info("ROOT AGENT: Routing to query_agent")
        return {"next_agent": "query_agent"}
    else:
        logger.info("ROOT AGENT: Answering directly")
        final_response = root_llm.invoke([{"role": "system", "content": system_prompt}] + list(messages))
        boxed_log(f"ROOT AGENT FINAL DIRECT RESPONSE: {final_response.content}", logger, level="info")
        return {"messages": [final_response], "next_agent": "__end__"}

def query_agent_node(state: AgentState) -> AgentState:
    """Query agent with system prompt"""
    logger.info("QUERY AGENT NODE ENTERED")
    # Inject system prompt into messages
    system_message = {
        "role": "system", 
        "content": f"{query_profile.instruction}\n\n{query_profile.description}"
    }
    messages_with_system = [system_message] + list(state["messages"])
    
    # Update state for agent
    agent_state = {**state, "messages": messages_with_system}
    result = query_agent.invoke(agent_state)

    # Log last AI message from query agent
    for msg in reversed(result['messages']):
        if hasattr(msg, 'content'):
            logger.info(f"QUERY AGENT LAST RESPONSE: {str(msg.content)[:200]}...")
            break
    
    return {"messages": result["messages"], "next_agent": "process_query_result"}

def route_after_root(state: AgentState) -> Literal["query_agent", "__end__"]:
    next_agent = state.get("next_agent", "__end__")
    # If next_agent is anything other than "query_agent", go to end
    if next_agent == "query_agent":
        return "query_agent"
    else:
        return "__end__"

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("root_agent", root_agent_node)
workflow.add_node("query_agent", query_agent_node)

workflow.set_entry_point("root_agent")
workflow.add_conditional_edges(
    "root_agent", 
    route_after_root,
    {"query_agent": "query_agent", "__end__": END}
)
workflow.add_edge("query_agent", "root_agent")

graph = workflow.compile()