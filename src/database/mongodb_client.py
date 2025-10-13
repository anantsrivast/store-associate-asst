

from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore, VectorIndexConfig
from langchain_openai import OpenAIEmbeddings
from src.config import config
from src.database.schemas import MongoDBSchemas
import logging

logger = logging.getLogger(__name__)


class MongoDBClientManager:
    """
    Manages MongoDB connections and provides access to collections,
    checkpointer, and store.

    This is a singleton class that maintains a single MongoDB connection
    throughout the application lifecycle.

    Usage:
        # Get the singleton instance
        db_manager = MongoDBClientManager()

        # Get collections
        customers = db_manager.get_collection("customers")

        # Get LangGraph components
        checkpointer = db_manager.get_checkpointer()
        store = db_manager.get_store()
    """

    _instance: Optional['MongoDBClientManager'] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    def __new__(cls):
        """Singleton pattern to ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the MongoDB client if not already initialized"""
        if self._client is None:
            self._connect()

    def _connect(self):
        """
        Establish connection to MongoDB.

        Creates the client, selects the database, and initializes
        all required collections with proper schemas and indexes.
        """
        try:
            logger.info(f"Connecting to MongoDB at {config.mongodb.uri}")

            # Create MongoDB client with connection pooling
            self._client = MongoClient(
                config.mongodb.uri,
                maxPoolSize=50,  # Maximum concurrent connections
                minPoolSize=10,  # Minimum idle connections
                serverSelectionTimeoutMS=5000,  # 5 second timeout
            )

            # Select database
            self._db = self._client[config.mongodb.db_name]

            # Test connection
            self._client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def setup_collections(self):
        """
        Create all collections with proper schemas and indexes.

        This should be called once during application setup.
        It's idempotent - safe to call multiple times.
        """
        try:
            # Setup customers collection
            self._setup_customers_collection()

            # Setup products collection
            self._setup_products_collection()

            # Setup purchases collection
            self._setup_purchases_collection()

            # Setup memories collection (for LangGraph Store)
            self._setup_memories_collection()

            logger.info("All collections setup successfully")

        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
            raise

    def _setup_customers_collection(self):
        """Setup customers collection with schema and indexes"""
        collection_name = config.mongodb.customers_collection

        # Create collection if it doesn't exist
        if collection_name not in self._db.list_collection_names():
            self._db.create_collection(
                collection_name,
                **MongoDBSchemas.get_customer_schema()
            )
            logger.info(f"Created {collection_name} collection")

        # Create indexes
        collection = self._db[collection_name]
        collection.create_indexes(MongoDBSchemas.get_customer_indexes())
        logger.info(f"Created indexes for {collection_name}")

    def _setup_products_collection(self):
        """Setup products collection with schema and indexes"""
        collection_name = config.mongodb.products_collection

        if collection_name not in self._db.list_collection_names():
            self._db.create_collection(
                collection_name,
                **MongoDBSchemas.get_product_schema()
            )
            logger.info(f"Created {collection_name} collection")

        collection = self._db[collection_name]
        collection.create_indexes(MongoDBSchemas.get_product_indexes())
        logger.info(f"Created indexes for {collection_name}")

    def _setup_purchases_collection(self):
        """Setup purchases collection with schema and indexes"""
        collection_name = config.mongodb.purchases_collection

        if collection_name not in self._db.list_collection_names():
            self._db.create_collection(
                collection_name,
                **MongoDBSchemas.get_purchase_schema()
            )
            logger.info(f"Created {collection_name} collection")

        collection = self._db[collection_name]
        collection.create_indexes(MongoDBSchemas.get_purchase_indexes())
        logger.info(f"Created indexes for {collection_name}")

    def _setup_memories_collection(self):
        """
        Setup customer_memories collection for LangGraph Store.

        This collection stores long-term memories with vector embeddings
        for semantic search.
        """
        collection_name = config.mongodb.memories_collection

        # Create collection (no schema validation for flexibility)
        if collection_name not in self._db.list_collection_names():
            self._db.create_collection(collection_name)
            logger.info(f"Created {collection_name} collection")

        collection = self._db[collection_name]

        # Create TTL index for automatic memory expiration
        collection.create_indexes([MongoDBSchemas.get_memories_ttl_index()])
        logger.info(f"Created TTL index for {collection_name}")

        # Note: Vector search index must be created in MongoDB Atlas UI
        # or via Atlas API. It cannot be created via pymongo.
        logger.info(
            f"Remember to create Atlas Vector Search index on {collection_name}"
        )

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        return self._db[collection_name]

    def get_checkpointer(self) -> MongoDBSaver:
        """
        Get LangGraph checkpointer for short-term memory.

        The checkpointer persists conversation state, enabling:
        - Conversation history within a session
        - Pause and resume conversations
        - Human-in-the-loop workflows
        - Time travel debugging

        Returns:
            MongoDBSaver instance configured with MongoDB
        """
        return MongoDBSaver.from_conn_string(
            conn_string=config.mongodb.uri,
            db_name=config.mongodb.db_name,
            collection_name=config.mongodb.checkpoints_collection
        )

    def get_store(self) -> MongoDBStore:
        from langchain_openai import OpenAIEmbeddings

        collection = self.get_collection(config.mongodb.memories_collection)

        # Only configure embeddings if we have an OpenAI key
        if config.llm.openai_api_key:
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small"
                )
                # Create index config with proper structure
                from langgraph.store.mongodb import VectorIndexConfig
                index_config = VectorIndexConfig(
                    dims=config.memory.embedding_dims,
                    embed=embeddings
                )
            except Exception as e:
                logger.warning(f"Could not configure embeddings: {e}")
                index_config = None
        else:
            index_config = None

        return MongoDBStore(
            collection=collection,
            index_config=index_config
        )

    def close(self):
        """Close the MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


# Global instance
db_manager = MongoDBClientManager()
