from langgraph.graph import StateGraph, END
from typing import TypedDict
from backend.agents.math_agent import MathAgent
from backend.agents.science_agent import ScienceAgent
from backend.agents.orchestrator_agent import OrchestratorAgent
import logging

logger = logging.getLogger(__name__)

class ChatState(TypedDict):
    query: str
    decision: str
    response: str

# Define states
def orchestrator_agent_node(state: ChatState) -> ChatState:
    query = state["query"]
    decision = OrchestratorAgent.decide(query)
    state["decision"] = decision
    logger.info(f"Orchestrator decision: {decision} for query: {query}")
    return state

def math_agent_node(state: ChatState) -> ChatState:
    state["response"] = MathAgent.handle(state["query"])
    logger.info(f"MathAgent response generated for query: {state['query']}")
    return state

def science_agent_node(state: ChatState) -> ChatState:
    state["response"] = ScienceAgent.handle(state["query"])
    logger.info(f"ScienceAgent response generated for query: {state['query']}")
    return state

def general_node(state: ChatState) -> ChatState:
    state["response"] = OrchestratorAgent.handle_general(state["query"])
    logger.info(f"General response generated for query: {state['query']}")
    return state

# Create LangGraph workflow
graph = StateGraph(ChatState)

# Nodes
graph.add_node("OrchestratorAgent", orchestrator_agent_node)
graph.add_node("MathAgent", math_agent_node)
graph.add_node("ScienceAgent", science_agent_node)
graph.add_node("General", general_node)

# Entry point
graph.set_entry_point("OrchestratorAgent")

# Conditional routing based on orchestrator decision
graph.add_conditional_edges(
    "OrchestratorAgent",
    lambda state: state["decision"],
    {
        "mathagent": "MathAgent",
        "scienceagent": "ScienceAgent",
        "general": "General"
    }
)

# All sub-agents nodes end the flow
graph.add_edge("MathAgent", END)
graph.add_edge("ScienceAgent", END)
graph.add_edge("General", END)

# Compile the graph
chat_graph = graph.compile()

def process_query(query: str) -> str:
    result = chat_graph.invoke({"query": query})
    return result.get("response", "No response generated.")