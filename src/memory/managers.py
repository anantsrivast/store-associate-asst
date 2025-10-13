# ============================================================================
# FILE: src/memory/managers.py
# DESCRIPTION: LangMem memory manager configurations
# ============================================================================

from langmem import create_memory_manager
from langchain.chat_models import init_chat_model
from src.memory.models import (
    CustomerPreference,
    ConversationEpisode,
    ConsolidatedInsight
)
from src.config import config
import logging

logger = logging.getLogger(__name__)


class MemoryManagers:
    """
    Factory class for creating and configuring LangMem memory managers.
    
    Memory managers are responsible for extracting structured information
    from conversations. They use LLMs to analyze text and extract relevant
    memories according to predefined schemas.
    
    There are three types of managers:
    1. Semantic Memory Manager: Extracts facts and preferences
    2. Episode Memory Manager: Summarizes conversations
    3. Consolidation Manager: Extracts patterns from multiple episodes
    """
    
    _semantic_manager = None
    _episode_manager = None
    _consolidation_manager = None
    
    @classmethod
    def get_semantic_manager(cls):
        """
        Get the semantic memory manager.
        
        This manager extracts individual facts and preferences from
        conversations in real-time (hot path). It's called during
        active conversations to capture important details.
        
        Input: Recent conversation messages
        Output: List of CustomerPreference objects
        
        Returns:
            Configured memory manager for semantic memory extraction
        """
        if cls._semantic_manager is None:
            logger.info("Initializing semantic memory manager")
            
            # Initialize the LLM for memory extraction
            llm_model = init_chat_model(config.llm.model)
            
            # Create the memory manager with specific instructions
            cls._semantic_manager = create_memory_manager(
                model=llm_model,
                schemas=[CustomerPreference],
                instructions="""
                You are a memory extraction specialist for a retail store.
                
                Your job is to extract customer preferences and facts from conversations.
                
                Focus on extracting:
                - Product preferences (brands, styles, colors)
                - Size information (clothing, shoes, etc.)
                - Budget and price sensitivity
                - Shopping occasions (gifts, personal, events)
                - Communication preferences
                - Activity or lifestyle information
                - Quality vs. price preferences
                
                Guidelines:
                - Only extract clear, explicit information
                - Set confidence based on how explicitly stated the preference is
                - Don't infer too much - stick to what was actually said
                - Extract multiple preferences if present
                - Update existing preferences if new information is provided
                
                Examples:
                - "I need size 8 shoes" → preference_type: "shoe_size", value: "8"
                - "I love Nike products" → preference_type: "favorite_brand", value: "Nike"
                - "My budget is around $100-150" → preference_type: "budget_range", value: "$100-150"
                """
            )
            
        return cls._semantic_manager
    
    @classmethod
    def get_episode_manager(cls):
        """
        Get the episode memory manager.
        
        This manager creates structured summaries of complete conversations.
        It's called when a conversation ends (background process) to create
        a searchable episode summary.
        
        Input: Full conversation history
        Output: ConversationEpisode object
        
        Returns:
            Configured memory manager for episode creation
        """
        if cls._episode_manager is None:
            logger.info("Initializing episode memory manager")
            
            llm_model = init_chat_model(config.llm.model)
            
            cls._episode_manager = create_memory_manager(
                model=llm_model,
                schemas=[ConversationEpisode],
                instructions="""
                You are a conversation summarization specialist.
                
                Your job is to create concise, informative summaries of customer
                interactions that can be searched later.
                
                Create summaries that include:
                
                1. SUMMARY (2-3 sentences):
                   - What the customer was looking for
                   - What was discussed
                   - How it ended
                
                2. CUSTOMER NEEDS:
                   - List the specific products or categories they wanted
                   - Include any constraints (size, price, occasion)
                
                3. PRODUCTS DISCUSSED:
                   - Specific products shown or mentioned
                   - Use product names or descriptions
                
                4. OUTCOME:
                   - Purchase made, browsing only, will return later, etc.
                   - Include specific products purchased if applicable
                
                5. KEY INSIGHTS:
                   - Important learnings about the customer
                   - Preferences revealed
                   - Shopping patterns observed
                
                6. SENTIMENT:
                   - Customer's emotional state
                   - Choose: "positive", "neutral", "negative", "frustrated", "excited"
                
                Guidelines:
                - Be concise but informative
                - Focus on details that will help in future interactions
                - Capture the essence of the interaction
                - Use specific product names and categories
                - Note any objections or concerns raised
                
                Example:
                "Customer Sarah visited looking for running shoes for marathon training.
                Discussed Nike Pegasus 40 in size 8. Customer was preparing for first
                marathon and needed guidance on proper running shoes. Purchased Nike
                Pegasus 40 and compression socks after detailed consultation."
                """
            )
            
        return cls._episode_manager
    
    @classmethod
    def get_consolidation_manager(cls):
        """
        Get the consolidation memory manager.
        
        This manager analyzes multiple episodes to extract behavioral patterns.
        It's run periodically (e.g., nightly) to discover higher-level insights.
        
        Input: Multiple episode summaries
        Output: List of ConsolidatedInsight objects
        
        Returns:
            Configured memory manager for pattern extraction
        """
        if cls._consolidation_manager is None:
            logger.info("Initializing consolidation memory manager")
            
            llm_model = init_chat_model(config.llm.model)
            
            cls._consolidation_manager = create_memory_manager(
                model=llm_model,
                schemas=[ConsolidatedInsight],
                instructions="""
                You are a pattern recognition specialist analyzing customer behavior.
                
                Your job is to identify meaningful patterns from multiple customer
                interactions over time.
                
                Look for patterns such as:
                
                1. SHOPPING FREQUENCY:
                   - How often does the customer shop?
                   - Are there seasonal patterns?
                   - Regular intervals (weekly, monthly, quarterly)?
                
                2. PRODUCT PREFERENCES:
                   - Consistent brand loyalty?
                   - Category preferences (always athletic wear, always casual)?
                   - Quality vs. price orientation?
                
                3. BEHAVIORAL PATTERNS:
                   - Time of day/week preferences?
                   - Purchase triggers (events, seasons, sales)?
                   - Browsing vs. buying behavior?
                   - Gift shopping patterns?
                
                4. DECISION-MAKING STYLE:
                   - Quick decisions or needs time?
                   - Price-sensitive or quality-focused?
                   - Influenced by recommendations or independent?
                
                5. SIZE AND FIT CONSISTENCY:
                   - Consistent sizing needs?
                   - Specific fit preferences?
                
                Guidelines:
                - Only identify patterns with at least 3 supporting episodes
                - Set confidence based on consistency and frequency
                - Be specific - "shops quarterly for running gear" not "shops sometimes"
                - Include timeframes where relevant
                - Note contradictions or changes in behavior
                - Focus on actionable insights for future interactions
                
                Examples of good patterns:
                - "Customer consistently purchases Nike athletic wear in size M every 3-4 months, 
                   typically before major races or training cycles. Budget range $100-150 per item."
                - "Customer is highly brand-loyal (Nike 90% of purchases) but price-sensitive, 
                   waits for sales or promotions before buying."
                - "Customer shops for both personal use (athletic) and gifts (casual wear), 
                   with gift shopping concentrated in November-December."
                """
            )
            
        return cls._consolidation_manager


