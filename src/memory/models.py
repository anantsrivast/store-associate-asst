
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class CustomerPreference(BaseModel):
    """
    Represents a single customer preference or fact (Semantic Memory).
    
    These are individual facts learned about the customer through
    conversations, stored in the semantic memory namespace.
    
    Examples:
        - Shoe size: 8
        - Favorite brand: Nike
        - Budget range: $100-150
        - Communication style: Prefers detailed explanations
    """
    preference_type: str = Field(
        description="Type of preference (e.g., 'size', 'brand', 'budget')"
    )
    value: str = Field(
        description="The preference value"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in this preference (0-1)"
    )
    source: str = Field(
        default="conversation",
        description="How this preference was learned"
    )
    first_observed: datetime = Field(
        default_factory=datetime.now,
        description="When this preference was first observed"
    )
    last_confirmed: datetime = Field(
        default_factory=datetime.now,
        description="Last time this preference was confirmed"
    )
    times_observed: int = Field(
        default=1,
        description="Number of times this preference was observed"
    )


class ConversationEpisode(BaseModel):
    """
    Represents a summary of a complete conversation (Episodic Memory).
    
    Episodes are created when a conversation session ends. They provide
    a concise summary that can be semantically searched later.
    
    This is stored in the episodic memory namespace with vector embeddings.
    """
    date: str = Field(
        description="Date of the interaction (ISO format)"
    )
    summary: str = Field(
        description="2-3 sentence summary of the conversation"
    )
    customer_needs: List[str] = Field(
        description="What the customer was looking for"
    )
    products_discussed: List[str] = Field(
        description="Products mentioned or shown"
    )
    outcome: str = Field(
        description="How the interaction ended (purchase, browsing, etc.)"
    )
    key_insights: str = Field(
        description="Important learnings about the customer"
    )
    sentiment: str = Field(
        description="Customer's emotional state (positive, neutral, negative)"
    )
    duration_minutes: Optional[int] = Field(
        default=None,
        description="Duration of the conversation"
    )
    associate_id: Optional[str] = Field(
        default=None,
        description="ID of the associate who helped"
    )


class ConsolidatedInsight(BaseModel):
    """
    Represents a pattern extracted from multiple episodes (Consolidated Memory).
    
    These are higher-level insights derived from analyzing multiple
    interactions. They represent learned patterns about customer behavior.
    
    Examples:
        - "Customer shops quarterly for running gear"
        - "Always buys Nike products in size 8"
        - "Budget-conscious, waits for sales"
    """
    pattern: str = Field(
        description="The discovered behavioral pattern"
    )
    evidence: List[str] = Field(
        description="Episode IDs that support this pattern"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this pattern (0-1)"
    )
    first_observed: str = Field(
        description="Date when pattern first emerged"
    )
    last_observed: str = Field(
        description="Most recent observation of this pattern"
    )
    frequency: int = Field(
        description="Number of times this pattern was observed"
    )
    pattern_type: str = Field(
        default="behavioral",
        description="Type of pattern (behavioral, seasonal, preference, etc.)"
    )


class CustomerProfile(BaseModel):
    """
    Complete customer profile combining all memory types.
    
    This is a convenience model that aggregates information from
    different memory namespaces for easy access.
    """
    customer_id: str
    name: str
    email: str
    phone: Optional[str] = None
    loyalty_tier: str = "bronze"
    
    # Semantic memory (facts and preferences)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Recent episodes
    recent_episodes: List[ConversationEpisode] = Field(default_factory=list)
    
    # Consolidated insights
    insights: List[ConsolidatedInsight] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None
    total_interactions: int = 0
    total_purchases: int = 0
    lifetime_value: float = 0.0


class RunningSummary(BaseModel):
    """
    Represents a rolling summary of a conversation.
    
    Used by the summarization node to compress long conversations
    while maintaining context.
    """
    summary: str = Field(
        description="Current summary of the conversation"
    )
    message_count: int = Field(
        description="Number of messages summarized"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now
    )
