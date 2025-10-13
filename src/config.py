
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Explicitly load .env file before anything else
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


class MemoryConfig(BaseSettings):
    """Configuration for memory-related settings."""
    
    ttl_seconds: int = Field(
        default=7776000,  # 90 days
        description="How long memories persist before auto-deletion"
    )
    
    summarization_threshold: int = Field(
        default=2000,
        description="Token count that triggers rolling summarization"
    )
    
    max_summary_tokens: int = Field(
        default=150,
        description="Maximum length of conversation summaries"
    )
    
    keep_recent_messages: int = Field(
        default=5,
        description="Recent messages to keep during summarization"
    )
    
    embedding_dims: int = Field(
        default=1536,
        description="Dimension of embedding vectors"
    )
    
    class Config:
        env_prefix = "MEMORY_"


class MongoDBConfig(BaseSettings):
    """MongoDB connection and collection configuration."""
    
    uri: str = Field(
        default="mongodb://localhost:27017/",
        description="MongoDB connection string"
    )
    
    database: str = Field(
        default="store_assistant",
        description="MongoDB database name"
    )
    
    checkpoints_collection: str = Field(
        default="checkpoints",
        description="Collection for conversation checkpoints"
    )
    
    memories_collection: str = Field(
        default="customer_memories",
        description="Collection for long-term memories"
    )
    
    customers_collection: str = Field(
        default="customers",
        description="Collection for customer profiles"
    )
    
    products_collection: str = Field(
        default="products",
        description="Collection for product catalog"
    )
    
    purchases_collection: str = Field(
        default="purchases",
        description="Collection for purchase history"
    )
    
    class Config:
        env_prefix = "MONGODB_"


class LLMConfig(BaseSettings):
    """LLM and embedding model configuration."""
    
    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="LLM model identifier"
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model identifier"
    )
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    temperature: float = Field(
        default=0.7,
        description="Temperature for LLM responses"
    )
    
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in LLM response"
    )


class AppConfig(BaseSettings):
    """Application-level configuration."""
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    app_name: str = Field(
        default="Store Associate Assistant",
        description="Application name"
    )
    
    class Config:
        env_prefix = "APP_"


class Config:
    """
    Main configuration singleton that aggregates all config sections.
    
    Usage:
        from src.config import config
        
        # Access MongoDB settings
        print(config.mongodb.uri)
        
        # Access memory settings
        print(config.memory.ttl_seconds)
    """
    
    def __init__(self):
        # dotenv is already loaded at module level
        # Initialize sub-configurations
        self.mongodb = MongoDBConfig()
        self.memory = MemoryConfig()
        self.llm = LLMConfig()
        self.app = AppConfig()
    
    def validate(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If required configuration is missing
        """
        # Check MongoDB URI
        if not self.mongodb.uri:
            raise ValueError("MONGODB_URI is required")
        
        # Check at least one API key is present
        if not self.llm.openai_api_key and not self.llm.anthropic_api_key:
            raise ValueError("Either OPENAI_API_KEY or ANTHROPIC_API_KEY is required")
        
        return True


# Global configuration instance
config = Config()