
import pytest
from src.memory.models import CustomerPreference, ConversationEpisode
from datetime import datetime


def test_customer_preference_creation():
    """Test creating a CustomerPreference model"""
    pref = CustomerPreference(
        preference_type="shoe_size",
        value="8",
        confidence=1.0
    )
    
    assert pref.preference_type == "shoe_size"
    assert pref.value == "8"
    assert pref.confidence == 1.0
    assert pref.times_observed == 1


def test_conversation_episode_creation():
    """Test creating a ConversationEpisode model"""
    episode = ConversationEpisode(
        date=datetime.now().isoformat(),
        summary="Customer visited looking for running shoes",
        customer_needs=["running shoes"],
        products_discussed=["Nike Pegasus"],
        outcome="Purchase completed",
        key_insights="Training for marathon",
        sentiment="positive"
    )
    
    assert "running shoes" in episode.customer_needs
    assert episode.sentiment == "positive"


# ============================================================================
# ADDITIONAL FILE: tests/test_agent.py
# DESCRIPTION: Basic agent tests
# ============================================================================

import pytest
from src.agent.state import AgentState
from langchain_core.messages import HumanMessage


def test_agent_state_creation():
    """Test creating agent state"""
    state = AgentState(
        messages=[HumanMessage(content="Hello")],
        customer_id="test_001",
        session_active=True,
        needs_summarization=False,
        context={},
        metadata={}
    )
    
    assert state["customer_id"] == "test_001"
    assert state["session_active"] == True
    assert len(state["messages"]) == 1



