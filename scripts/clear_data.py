# ============================================================================
# FILE: scripts/clear_data.py
# DESCRIPTION: Clear all data from MongoDB
# ============================================================================

#!/usr/bin/env python3

from src.database.mongodb_client import db_manager
from src.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_all_data():
    """
    Clear all data from MongoDB collections.
    
    WARNING: This will delete all data!
    """
    try:
        logger.warning("⚠️  This will delete ALL data from the database!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        
        if response.lower() != 'yes':
            logger.info("Cancelled.")
            return
        
        logger.info("Clearing all data...")
        
        # Get all collections
        collections = [
            config.mongodb.customers_collection,
            config.mongodb.products_collection,
            config.mongodb.purchases_collection,
            config.mongodb.memories_collection,
            config.mongodb.checkpoints_collection,
            "checkpoint_writes"  # LangGraph creates this automatically
        ]
        
        # Delete all documents from each collection
        for coll_name in collections:
            try:
                collection = db_manager.get_collection(coll_name)
                result = collection.delete_many({})
                logger.info(f"✓ Cleared {result.deleted_count} documents from {coll_name}")
            except Exception as e:
                logger.warning(f"Could not clear {coll_name}: {e}")
        
        logger.info("✓ All data cleared!")
        logger.info("Run 'python -m src.data.seed_database' to regenerate data")
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")


if __name__ == "__main__":
    clear_all_data()
