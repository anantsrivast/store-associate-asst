
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


