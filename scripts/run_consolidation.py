# ============================================================================
# FILE: scripts/run_consolidation.py
# DESCRIPTION: Manually trigger memory consolidation
# ============================================================================

#!/usr/bin/env python3

import asyncio
import sys
from src.memory.consolidation import MemoryConsolidator
from src.database.mongodb_client import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_consolidation(customer_id: str = None):
    """
    Run memory consolidation.
    
    Args:
        customer_id: If provided, consolidate for specific customer.
                    Otherwise, consolidate for all customers.
    """
    try:
        store = db_manager.get_store()
        consolidator = MemoryConsolidator(store)
        
        if customer_id:
            logger.info(f"Running consolidation for customer: {customer_id}")
            insights = await consolidator.consolidate_customer_memories(customer_id)
            logger.info(f"Created {len(insights)} new insights")
        else:
            logger.info("Running consolidation for all active customers")
            await consolidator.consolidate_all_active_customers()
        
        logger.info("✓ Consolidation complete!")
        
    except Exception as e:
        logger.error(f"Error running consolidation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    customer_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_consolidation(customer_id))


