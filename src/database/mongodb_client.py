from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langgraph.store.mongodb.base import create_vector_index_config
from src.config import config
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MongoDBClientManager:
    """
    Manages MongoDB connections for the Store Associate Agent.

    Provides:
    - Database and collection access
    - MongoDBSaver for conversation checkpoints (short-term memory)
    - MongoDBStore for long-term memory with vector search
    - Vector store for semantic search
    """

    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._checkpointer: Optional[MongoDBSaver] = None
        self._store: Optional[MongoDBStore] = None
        self._vector_store: Optional[MongoDBAtlasVectorSearch] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

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

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get embeddings model for vector operations"""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=config.llm.openai_api_key
            )
        return self._embeddings

    def get_checkpointer(self) -> MongoDBSaver:
        """
        Get LangGraph checkpointer for conversation state (short-term memory).

        This stores the current conversation messages and enables:
        - Conversation resumption
        - Time travel debugging
        - Human-in-the-loop workflows
        """
        if self._checkpointer is None:
            client = self.get_client()
            self._checkpointer = MongoDBSaver(
                client,
                db_name=config.mongodb.database,
                collection_name=config.mongodb.checkpoints_collection
            )
            logger.info("MongoDBSaver checkpointer initialized")
        return self._checkpointer
    def get_store(self) -> MongoDBStore:
        """
        Get LangGraph store for long-term memory with proper embedding configuration.
        """
        if self._store is None:
            from langchain_openai import OpenAIEmbeddings
            from langgraph.store.mongodb.base import create_vector_index_config
            from langchain_core.embeddings import Embeddings
            
            collection = self.get_collection(config.mongodb.memories_collection)
            
            # Configure base embeddings
            base_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=config.llm.openai_api_key
            )
            
            # Create a wrapper class that properly inherits
            class DebugEmbeddings(Embeddings):
                def __init__(self, base):
                    self.base = base
                
                def embed_documents(self, texts):
                    logger.info(f"=== EMBEDDING DEBUG ===")
                    logger.info(f"embed_documents called")
                    logger.info(f"Input type: {type(texts)}")
                    logger.info(f"Input: {texts}")
                    logger.info(f"Length: {len(texts) if hasattr(texts, '__len__') else 'N/A'}")
                    
                    if not texts:
                        logger.error("EMPTY TEXTS LIST!")
                        return []
                    
                    result = self.base.embed_documents(texts)
                    logger.info(f"Result length: {len(result)}")
                    logger.info(f"=== END EMBEDDING DEBUG ===")
                    return result
                
                def embed_query(self, text):
                    logger.info(f"=== EMBEDDING DEBUG (query) ===")
                    logger.info(f"Input: {text}")
                    result = self.base.embed_query(text)
                    logger.info(f"Result length: {len(result)}")
                    logger.info(f"=== END EMBEDDING DEBUG ===")
                    return result
            
            embeddings = DebugEmbeddings(base_embeddings)
            
            # Create index config
            index_config = create_vector_index_config(
                embed=embeddings,
                dims=1536,
                fields=["content"],
                name="vector_index",
                relevance_score_fn="cosine",
                embedding_key="embedding",
                filters=["namespace", "key"]
            )
            
            self._store = MongoDBStore(
                collection=collection,
                index_config=index_config
            )
            
            logger.info("MongoDBStore initialized with vector search capabilities")
        
        return self._store
    def get_vector_store(self) -> MongoDBAtlasVectorSearch:
        """
        Get LangChain MongoDB Atlas Vector Store for semantic search.

        This provides:
        - Vector embeddings storage
        - Semantic similarity search
        - Full integration with LangChain

        Returns:
            MongoDBAtlasVectorSearch instance
        """
        if self._vector_store is None:
            collection = self.get_collection(
                config.mongodb.memories_collection)
            embeddings = self.get_embeddings()

            self._vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name="vector_index",
                text_key="value.content",  # Field containing the text to search
                embedding_key="embedding"  # Field containing the vector
            )

            logger.info(
                "MongoDBAtlasVectorSearch initialized with vector search")

        return self._vector_store

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
            try:
                products_collection.create_index([
                    ("name", "text"),
                    ("description", "text"),
                    ("brand", "text")
                ], name="product_search_index")
                logger.info("Created indexes for products")
            except Exception as e:
                logger.debug(f"Product index may already exist: {e}")

            # Customers collection - index on customer_id
            customers_collection = db[config.mongodb.customers_collection]
            try:
                customers_collection.create_index("customer_id", unique=True)
                logger.info("Created indexes for customers")
            except Exception as e:
                logger.debug(f"Customer index may already exist: {e}")

            # Purchases collection - indexes for queries
            purchases_collection = db[config.mongodb.purchases_collection]
            try:
                purchases_collection.create_index("customer_id")
                purchases_collection.create_index("purchase_date")
                logger.info("Created indexes for purchases")
            except Exception as e:
                logger.debug(f"Purchase indexes may already exist: {e}")

            # Memories collection - compound index for namespace lookups
            memories_collection = db[config.mongodb.memories_collection]
            try:
                memories_collection.create_index([
                    ("customer_id", 1),
                    ("timestamp", -1)
                ])
                logger.info("Created indexes for memories")
            except Exception as e:
                logger.debug(f"Memory indexes may already exist: {e}")

            logger.info("All collections setup successfully")

        except Exception as e:
            logger.warning(f"Error setting up collections: {e}")

    def close(self):
        """Close the MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


# Global instance
db_manager = MongoDBClientManager()
