"""
LangGraph workflow for store assistant agent.

This module defines the graph structure and routing logic.
All node implementations are in nodes.py for better organization.
"""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.database.mongodb_client import db_manager
from src.agent.nodes import (
    check_summarization_node,
    summarize_conversation_node,
    agent_node,
    extract_semantic_memories_node,
    create_episode_node
)
from langchain_core.messages import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


def route_after_summarization(state: AgentState) -> str:
    """
    Route after checking summarization.
    Returns: "summarize" or "agent"
    """
    if state.get("needs_summarization", False):
        logger.info("Routing to summarize node")
        return "summarize"
    logger.info("Routing directly to agent node")
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """
    Route after agent processing.
    Returns: "create_episode" or "extract_memories"
    """
    if not state.get("session_active", True):
        logger.info("Session ending - routing to create_episode")
        return "create_episode"
    logger.info("Session active - routing to extract_memories")
    return "extract_memories"


def create_agent_graph():
    """
    Create and compile the complete agent graph with full workflow.
    
    Graph Flow:
    START → check_summarization → (conditional)
        ├─> [needs_summarization=True] → summarize → agent
        └─> [needs_summarization=False] → agent
    
    agent → (conditional)
        ├─> [session_active=False] → create_episode → END
        └─> [session_active=True] → extract_memories → END
    
    Returns:
        Compiled graph ready for execution
    """
    logger.info("Building agent graph with full workflow")
    
    checkpointer = db_manager.get_checkpointer()
    store = db_manager.get_store()
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes (importing from nodes.py)
    workflow.add_node("check_summarization", check_summarization_node)
    workflow.add_node("summarize", summarize_conversation_node)
    workflow.add_node("agent", lambda s: agent_node(s, store))
    workflow.add_node("extract_memories", lambda s: extract_semantic_memories_node(s, store))
    workflow.add_node("create_episode", lambda s: create_episode_node(s, store))
    
    # Set entry point
    workflow.set_entry_point("check_summarization")
    
    # Conditional edge from check_summarization
    workflow.add_conditional_edges(
        "check_summarization",
        route_after_summarization,
        {
            "summarize": "summarize",
            "agent": "agent"
        }
    )
    
    # Edge from summarize to agent
    workflow.add_edge("summarize", "agent")
    
    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "create_episode": "create_episode",
            "extract_memories": "extract_memories"
        }
    )
    
    # Edges to END
    workflow.add_edge("extract_memories", END)
    workflow.add_edge("create_episode", END)
    
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Agent graph compiled successfully with full workflow")
    return graph


class StoreAssistantAgent:
    """
    High-level wrapper for the store assistant agent.
    """
    
    def __init__(self):
        """Initialize the agent with compiled graph"""
        self.graph = create_agent_graph()
        logger.info("StoreAssistantAgent initialized")
    
    def chat(
        self,
        customer_id: str,
        message: str,
        thread_id: str,
        session_active: bool = True
    ) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            customer_id: Customer identifier
            message: User's message
            thread_id: Conversation thread identifier
            session_active: Whether the session is still active
            
        Returns:
            Agent's response text
        """
        cfg = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        input_state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
            "session_active": session_active,
            "needs_summarization": False,
            "context": {},
            "metadata": {}
        }
        
        result = self.graph.invoke(input_state, cfg)
        
        result_messages = result["messages"]
        
        # Find the last meaningful AI response (skip tool-calling messages)
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage):
                # Skip intermediate tool-calling messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"Skipping AIMessage with tool_calls (intermediate message)")
                    continue
                
                # Found a text response
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    return msg.content
                elif hasattr(msg, 'content') and isinstance(msg.content, list):
                    # Handle content blocks
                    for block in msg.content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text = block.get('text', '').strip()
                            if text:
                                return text
        
        # No response found
        return "I apologize, but I couldn't generate a response."
    
    def end_session(self, customer_id: str, thread_id: str):
        """
        End the conversation session and create an episode summary.
        
        Args:
            customer_id: Customer identifier
            thread_id: Conversation thread identifier
        """
        logger.info(f"Ending session for thread {thread_id}")
        
        cfg = {"configurable": {"thread_id": thread_id}}
        
        # Get the current state from checkpoint
        current_state = self.graph.get_state(cfg)
        
        # Update only the session_active flag
        current_state.values["session_active"] = False
        
        # Invoke with the full state (including messages)
        self.graph.invoke(current_state.values, cfg)
        
        logger.info(f"Session ended and episode created")
