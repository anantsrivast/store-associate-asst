# ============================================================================
# FILE: src/memory/consolidation.py
# DESCRIPTION: Background memory consolidation logic
# ============================================================================

from typing import List
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langgraph.store.mongodb import MongoDBStore
from src.memory.managers import MemoryManagers
from src.memory.models import ConsolidatedInsight
import logging

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Handles background consolidation of memories.
    
    This class runs periodically (e.g., as a cron job) to analyze
    multiple episodes and extract behavioral patterns.
    
    The consolidation process:
    1. Fetch recent episodes for a customer
    2. Check if we have enough episodes to consolidate
    3. Extract episode summaries
    4. Use consolidation manager to identify patterns
    5. Store consolidated insights
    6. Optionally clean up old episodes
    """
    
    def __init__(self, store: MongoDBStore):
        """
        Initialize the consolidator.
        
        Args:
            store: MongoDBStore instance for accessing memories
        """
        self.store = store
        self.consolidation_manager = MemoryManagers.get_consolidation_manager()
    
    async def consolidate_customer_memories(
        self,
        customer_id: str,
        min_episodes: int = 5,
        lookback_days: int = 90
    ) -> List[ConsolidatedInsight]:
        """
        Consolidate memories for a single customer.
        
        Args:
            customer_id: Customer identifier
            min_episodes: Minimum number of episodes needed for consolidation
            lookback_days: How far back to look for episodes
            
        Returns:
            List of newly created consolidated insights
        """
        try:
            logger.info(f"Starting consolidation for customer {customer_id}")
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Fetch recent episodes from the store
            episodes = await self.store.asearch(
                namespace_prefix=("customers", customer_id, "episodes"),
                filter={"date": {"$gte": cutoff_date.isoformat()}},
                limit=50  # Limit to most recent 50 episodes
            )
            
            logger.info(f"Found {len(episodes)} episodes for customer {customer_id}")
            
            # Check if we have enough episodes
            if len(episodes) < min_episodes:
                logger.info(
                    f"Not enough episodes ({len(episodes)}) for consolidation. "
                    f"Minimum required: {min_episodes}"
                )
                return []
            
            # Extract episode summaries and metadata
            episode_texts = []
            for i, episode in enumerate(episodes):
                episode_data = episode.value
                # Format episode information for the LLM
                episode_text = f"""
                Episode {i+1} ({episode_data.get('date', 'unknown date')}):
                Summary: {episode_data.get('summary', 'N/A')}
                Needs: {', '.join(episode_data.get('customer_needs', []))}
                Products: {', '.join(episode_data.get('products_discussed', []))}
                Outcome: {episode_data.get('outcome', 'N/A')}
                Sentiment: {episode_data.get('sentiment', 'N/A')}
                """
                episode_texts.append(episode_text)
            
            # Combine all episodes into a single text
            combined_text = "\n---\n".join(episode_texts)
            
            logger.info(f"Analyzing {len(episodes)} episodes for patterns")
            
            # Extract consolidated insights using the manager
            insights = self.consolidation_manager.extract_memories(
                messages=[HumanMessage(content=combined_text)]
            )
            
            logger.info(f"Extracted {len(insights)} patterns")
            
            # Store each insight in the consolidated memory namespace
            stored_insights = []
            for insight in insights:
                # Create a unique key based on pattern content
                pattern_key = f"pattern_{hash(insight.pattern) % 10000:04d}"
                
                # Store in MongoDB via LangGraph store
                await self.store.aput(
                    namespace=("customers", customer_id, "insights"),
                    key=pattern_key,
                    value={
                        "pattern": insight.pattern,
                        "evidence": insight.evidence,
                        "confidence": insight.confidence,
                        "first_observed": insight.first_observed,
                        "last_observed": insight.last_observed,
                        "frequency": insight.frequency,
                        "pattern_type": insight.pattern_type,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                stored_insights.append(insight)
                logger.info(f"Stored insight: {insight.pattern[:50]}...")
            
            logger.info(
                f"Consolidation complete for customer {customer_id}. "
                f"Created {len(stored_insights)} new insights"
            )
            
            return stored_insights
            
        except Exception as e:
            logger.error(f"Error consolidating memories for {customer_id}: {e}")
            raise
    
    async def consolidate_all_active_customers(
        self,
        min_episodes: int = 5,
        lookback_days: int = 90
    ):
        """
        Consolidate memories for all customers with sufficient episodes.
        
        This is the main entry point for batch consolidation jobs.
        
        Args:
            min_episodes: Minimum episodes needed per customer
            lookback_days: Lookback period for episodes
        """
        try:
            logger.info("Starting batch consolidation for all active customers")
            
            # Get all customer IDs that have episodes
            # This requires querying MongoDB directly to find unique customer IDs
            # in the episodes namespace
            
            from src.database.mongodb_client import db_manager
            collection = db_manager.get_collection(
                db_manager.config.mongodb.memories_collection
            )
            
            # Find unique customer IDs with episodes
            pipeline = [
                {"$match": {"namespace.1": {"$exists": True}}},  # Has customer_id
                {"$group": {"_id": "$namespace.1"}},  # Group by customer_id
                {"$limit": 100}  # Process up to 100 customers per run
            ]
            
            customer_ids = [doc["_id"] for doc in collection.aggregate(pipeline)]
            
            logger.info(f"Found {len(customer_ids)} customers with memories")
            
            # Consolidate memories for each customer
            total_insights = 0
            for customer_id in customer_ids:
                insights = await self.consolidate_customer_memories(
                    customer_id=customer_id,
                    min_episodes=min_episodes,
                    lookback_days=lookback_days
                )
                total_insights += len(insights)
            
            logger.info(
                f"Batch consolidation complete. "
                f"Processed {len(customer_ids)} customers, "
                f"created {total_insights} total insights"
            )
            
        except Exception as e:
            logger.error(f"Error in batch consolidation: {e}")
            raise


