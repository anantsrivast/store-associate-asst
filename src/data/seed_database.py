
import asyncio
from src.database.mongodb_client import db_manager
from src.data.synthetic_data import SyntheticDataGenerator
from src.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_database(clear_existing: bool = False):
    """
    Seed the MongoDB database with synthetic data.
    
    Args:
        clear_existing: If True, clear existing data before seeding
    """
    try:
        logger.info("Starting database seeding")
        
        # Setup collections first
        db_manager.setup_collections()
        
        # Get collections
        customers_coll = db_manager.get_collection(config.mongodb.customers_collection)
        products_coll = db_manager.get_collection(config.mongodb.products_collection)
        purchases_coll = db_manager.get_collection(config.mongodb.purchases_collection)
        
        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing data")
            customers_coll.delete_many({})
            products_coll.delete_many({})
            purchases_coll.delete_many({})
        
        # Generate synthetic data
        generator = SyntheticDataGenerator(num_customers=50, num_products=200)
        
        customers = generator.generate_customers()
        products = generator.generate_products()
        purchases = generator.generate_purchases(customers, products)
        initial_memories = generator.generate_initial_memories(customers)
        
        # Insert data
        logger.info("Inserting customers")
        customers_coll.insert_many(customers)
        
        logger.info("Inserting products")
        products_coll.insert_many(products)
        
        logger.info("Inserting purchases")
        if purchases:
            purchases_coll.insert_many(purchases)
        
        # Insert initial memories
        logger.info("Inserting initial memories")
        store = db_manager.get_store()
        
        for customer_id, memories in initial_memories.items():
            for memory in memories:
                store.put(
                    namespace=memory["namespace"],
                    key=memory["key"],
                    value=memory["value"],
                    index=True if "episode" in memory["key"] else False
                )
        
        logger.info("Database seeding complete!")
        logger.info(f"  - {len(customers)} customers")
        logger.info(f"  - {len(products)} products")
        logger.info(f"  - {len(purchases)} purchases")
        logger.info(f"  - {sum(len(m) for m in initial_memories.values())} initial memories")
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise


if __name__ == "__main__":
    seed_database(clear_existing=True)
