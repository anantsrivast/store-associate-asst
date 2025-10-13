
#!/usr/bin/env python3

import sys
from src.database.mongodb_client import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mongodb():
    """
    Setup MongoDB collections and indexes.
    
    This script should be run once during initial setup.
    """
    try:
        logger.info("Setting up MongoDB...")
        
        # Test connection
        logger.info("Testing MongoDB connection...")
        client = db_manager._client
        client.admin.command('ping')
        logger.info("✓ MongoDB connection successful")
        
        # Setup collections
        logger.info("Creating collections and indexes...")
        db_manager.setup_collections()
        logger.info("✓ Collections and indexes created")
        
        # Remind about Atlas Vector Search index
        logger.info("\n" + "="*60)
        logger.info("IMPORTANT: Manual Step Required")
        logger.info("="*60)
        logger.info("\nYou must create an Atlas Vector Search index manually:")
        logger.info("\n1. Go to MongoDB Atlas Console")
        logger.info("2. Navigate to your cluster → 'Search' tab")
        logger.info("3. Click 'Create Search Index'")
        logger.info("4. Choose 'JSON Editor'")
        logger.info("5. Use this configuration:\n")
        logger.info("""
{
  "fields": [
    {
      "type": "vector",
      "path": "value.embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
        """)
        logger.info("\n6. Name the index: 'vector_index'")
        logger.info("7. Select collection: 'customer_memories'")
        logger.info("="*60)
        
        logger.info("\n✓ MongoDB setup complete!")
        
    except Exception as e:
        logger.error(f"Error setting up MongoDB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_mongodb()


