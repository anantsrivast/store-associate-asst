
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from src.agent.state import AgentState
from src.agent.nodes import (
    check_summarization_node,
    summarize_conversation_node,
    agent_node,
    extract_semantic_memories_node,
    create_episode_node
)
from src.database.mongodb_client import db_manager
import logging

logger = logging.getLogger(__name__)


def create_agent_graph():
    """
    Create and compile the complete agent graph.
    
    The graph flow:
    1. Entry → Check Summarization
    2. Check Summarization → (if needed) Summarize → Agent
    3. Check Summarization → (if not needed) → Agent
    4. Agent → Extract Memories
    5. Extract Memories → (if session active) → END
    6. Extract Memories → (if session ended) → Create Episode → END
    
    Returns:
        Compiled graph ready for execution
    """
    logger.info("Building agent graph")
    
    # Get MongoDB components
    checkpointer = db_manager.get_checkpointer()
    store = db_manager.get_store()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_summarization", check_summarization_node)
    workflow.add_node("summarize", summarize_conversation_node)
    workflow.add_node("agent", lambda s: agent_node(s, store))
    workflow.add_node("extract_memories", lambda s: extract_semantic_memories_node(s, store))
    workflow.add_node("create_episode", lambda s: create_episode_node(s, store))
    
    # Define edges
    # Entry point
    workflow.set_entry_point("check_summarization")
    
    # Conditional edge: summarize if needed
    workflow.add_conditional_edges(
        "check_summarization",
        lambda state: "summarize" if state.get("needs_summarization", False) else "agent",
        {
            "summarize": "summarize",
            "agent": "agent"
        }
    )
    
    # After summarization, go to agent
    workflow.add_edge("summarize", "agent")
    
    # After agent, extract memories
    workflow.add_edge("agent", "extract_memories")
    
    # Conditional edge: create episode if session ended
    workflow.add_conditional_edges(
        "extract_memories",
        lambda state: "create_episode" if not state.get("session_active", True) else "end",
        {
            "create_episode": "create_episode",
            "end": END
        }
    )
    
    # After episode creation, end
    workflow.add_edge("create_episode", END)
    
    # Compile the graph with checkpointer and store
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    logger.info("Agent graph compiled successfully")
    
    return graph


class StoreAssistantAgent:
    """
    High-level wrapper for the store assistant agent.
    
    This provides a clean interface for interacting with the agent,
    handling configuration and thread management.
    
    Usage:
        agent = StoreAssistantAgent()
        
        # Start conversation
        response = agent.chat(
            customer_id="sarah_123",
            message="Hi, I need running shoes",
            thread_id="thread_1"
        )
        
        # Continue conversation
        response = agent.chat(
            customer_id="sarah_123",
            message="Size 8 please",
            thread_id="thread_1"
        )
        
        # End session
        agent.end_session(thread_id="thread_1")
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
        from langchain_core.messages import HumanMessage
        
        # Create config with thread_id for checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Create input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
            "session_active": session_active,
            "needs_summarization": False,
            "context": {},
            "metadata": {}
        }
        
        # Invoke the graph
        result = self.graph.invoke(input_state, config)
        
        # Extract the last assistant message
        last_message = result["messages"][-1]
        
        return last_message.content
    
    def end_session(self, customer_id: str, thread_id: str):
        """
        End a conversation session and trigger episode creation.
        
        Args:
            customer_id: Customer identifier
            thread_id: Conversation thread identifier
        """
        # Send a final message with session_active=False
        # This will trigger episode creation
        self.chat(
            customer_id=customer_id,
            message="",  # Empty message
            thread_id=thread_id,
            session_active=False
        )
        
        logger.info(f"Session ended for thread {thread_id}")


