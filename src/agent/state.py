# ============================================================================
# FILE: src/agent/state.py
# DESCRIPTION: LangGraph state definitions
# ============================================================================

from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage
from src.memory.models import RunningSummary


class AgentState(TypedDict):
    """
    Main state for the LangGraph agent.
    
    This state is passed between nodes in the graph and contains
    all information needed for the conversation.
    
    Fields:
        messages: List of conversation messages (user + assistant)
        customer_id: Unique identifier for the current customer
        session_active: Whether the conversation session is still active
        needs_summarization: Flag indicating if conversation should be compressed
        context: Running summaries for context window management
        metadata: Additional metadata (associate_id, store_location, etc.)
    """
    # Core conversation state
    messages: List[BaseMessage]
    
    # Customer information
    customer_id: str
    
    # Session management
    session_active: bool
    
    # Summarization control
    needs_summarization: bool
    
    # Context compression
    context: Dict[str, RunningSummary]
    
    # Additional metadata
    metadata: Optional[Dict[str, any]]


class LLMInputState(TypedDict):
    """
    State passed to the LLM node after summarization.
    
    This is a lighter state that includes the compressed conversation
    instead of all raw messages.
    
    Fields:
        summarized_messages: Compressed conversation (recent + summary)
        context: Running summary context
        customer_id: Customer identifier
        metadata: Additional context
    """
    summarized_messages: List[BaseMessage]
    context: Dict[str, RunningSummary]
    customer_id: str
    metadata: Optional[Dict[str, any]]
