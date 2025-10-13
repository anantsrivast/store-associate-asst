from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from src.config import config
import logging

logger = logging.getLogger(__name__)


class MongoDBClientManager:
    """
    Manages MongoDB connections for the Store Associate Agent.
    
    Provides:
    - Database and collection access
    - MongoDBSaver for conversation checkpoints (short-term memory)
    - MongoDBStore for long-term memory with vector search
    """
    
    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._checkpointer: Optional[MongoDBSaver] = None
        self._store: Optional[MongoDBStore] = None
    
    def get_client(self) -> MongoClient:
        """Get or create MongoDB client"""
        if self._client is None:
            self._client = MongoClient(
                config.mongodb.uri,
                serverSelectionTimeoutMS=5000
            )
            logger.info("MongoDB client connected")
        return self._client
    
    def get_database(self) -> Database:
        """Get the main database"""
        if self._db is None:
            client = self.get_client()
            self._db = client[config.mongodb.database]
            logger.info(f"Connected to database: {config.mongodb.database}")
        return self._db
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a specific collection"""
        db = self.get_database()
        return db[collection_name]
    
    def get_checkpointer(self) -> MongoDBSaver:
        """
        Get LangGraph checkpointer for conversation state (short-term memory).
        
        This stores the current conversation messages and enables:
        - Conversation resumption
        - Time travel debugging
        - Human-in-the-loop workflows
        """
        if self._checkpointer is None:
            self._checkpointer = MongoDBSaver.from_conn_string(
                conn_string=config.mongodb.uri,
                db_name=config.mongodb.database,
                collection_name=config.mongodb.checkpoints_collection
            )
            logger.info("MongoDBSaver checkpointer initialized")
        return self._checkpointer
    
    def get_store(self) -> MongoDBStore:
        """
        Get LangGraph store for long-term memory with vector search.
        
        The store provides:
        - Namespace-based memory organization
        - Vector search for semantic retrieval
        - Cross-session persistence
        
        Returns:
            MongoDBStore instance configured with vector search
        """
        if self._store is None:
            from langchain_openai import OpenAIEmbeddings
            
            collection = self.get_collection(config.mongodb.memories_collection)
            
            # Configure embeddings for vector search
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=config.llm.openai_api_key
            )
            
            # Create index_config with all required fields
            index_config = {
                "fields": ["value"],  # Fields to index for vector search
                "dims": 1536,  # text-embedding-3-small produces 1536-dim vectors
                "embed": embeddings.embed_query,  # Embedding function
                "filters": []  # Additional filters (empty list)
            }
            
            # Initialize store with embeddings and increased timeout
            self._store = MongoDBStore(
                collection=collection,
                index_config=index_config,
                index_timeout=100  # Increased timeout to 100 seconds
            )
            
            logger.info("MongoDBStore initialized with vector search capabilities")
        
        return self._store
    
    def setup_collections(self):
        """
        Set up MongoDB collections with indexes.
        
        Creates necessary indexes for:
        - Text search on products
        - Customer lookups
        - Purchase history queries
        """
        try:
            db = self.get_database()
            
            # Products collection - text index for search
            products_collection = db[config.mongodb.products_collection]
            products_collection.create_index([
                ("name", "text"),
                ("description", "text"),
                ("brand", "text")
            ], name="product_search_index")
            
            # Customers collection - index on customer_id
            customers_collection = db[config.mongodb.customers_collection]
            customers_collection.create_index("customer_id", unique=True)
            
            # Purchases collection - indexes for queries
            purchases_collection = db[config.mongodb.purchases_collection]
            purchases_collection.create_index("customer_id")
            purchases_collection.create_index("purchase_date")
            
            logger.info("MongoDB collections and indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error setting up collections (may already exist): {e}")
    
    def close(self):
        """Close the MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


# Global instance
db_manager = MongoDBClientManager()