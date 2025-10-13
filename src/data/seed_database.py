from datetime import datetime
from src.database.mongodb_client import db_manager
from src.data.synthetic_data import SyntheticDataGenerator
from src.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_value(obj):
    """
    Recursively sanitize a value to ensure all fields are strings.
    
    This is necessary because the embedding function can only handle strings,
    not lists or other complex objects.
    
    Args:
        obj: The object to sanitize (dict, list, or primitive)
        
    Returns:
        Sanitized object with all lists converted to strings
    """
    if isinstance(obj, dict):
        return {k: sanitize_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Convert list to comma-separated string
        return ", ".join(str(item) for item in obj)
    elif obj is None:
        return ""
    else:
        return str(obj)


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
            # Also clear memories
            memories_coll = db_manager.get_collection(config.mongodb.memories_collection)
            memories_coll.delete_many({})
        
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
        
        # Insert initial memories with sanitization
        logger.info("Inserting initial memories")
        store = db_manager.get_store()
        
        total_memories = 0
        for customer_id, memories in initial_memories.items():
            for memory in memories:
                # Sanitize the value to ensure all fields are strings
                sanitized_value = sanitize_value(memory["value"])
                
                try:
                    store.put(
                        namespace=memory["namespace"],
                        key=memory["key"],
                        value=sanitized_value
                    )
                    total_memories += 1
                    logger.debug(f"Inserted memory: {memory['key']} for customer {customer_id}")
                except Exception as e:
                    logger.error(f"Error inserting memory {memory['key']}: {e}")
                    logger.error(f"Sanitized value was: {sanitized_value}")
                    # Continue with other memories even if one fails
                    continue
        
        logger.info(f"Inserted {total_memories} initial memories")
        logger.info("Database seeding complete!")
        logger.info(f"Summary:")
        logger.info(f"  - Customers: {len(customers)}")
        logger.info(f"  - Products: {len(products)}")
        logger.info(f"  - Purchases: {len(purchases)}")
        logger.info(f"  - Memories: {total_memories}")
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise


if __name__ == "__main__":
    seed_database(clear_existing=True)