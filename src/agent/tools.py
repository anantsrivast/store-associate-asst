# ============================================================================
# FILE: src/agent/tools.py
# DESCRIPTION: Tools available to the agent
# ============================================================================

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any
from src.database.mongodb_client import db_manager
from src.config import config
import logging

logger = logging.getLogger(__name__)


@tool
def search_products(query: str, category: str = None, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for products in the catalog.
    
    Use this tool when the customer asks about products or you need
    to find product information.
    
    Args:
        query: Search query (product name, keywords, description)
        category: Optional category filter (e.g., "shoes", "athletic_wear")
        max_results: Maximum number of results to return
        
    Returns:
        List of product dictionaries with details
    """
    try:
        collection = db_manager.get_collection(config.mongodb.products_collection)
        
        # Build search query
        search_filter = {}
        if category:
            search_filter["category"] = category
        
        # Text search on name and description
        if query:
            search_filter["$text"] = {"$search": query}
        
        # Execute search
        products = list(collection.find(
            search_filter,
            limit=max_results
        ).sort("price", 1))  # Sort by price ascending
        
        # Format results
        results = []
        for product in products:
            results.append({
                "sku": product.get("sku"),
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price"),
                "description": product.get("description", "")[:200],  # Truncate
                "variants": product.get("variants", [])
            })
        
        logger.info(f"Product search for '{query}' returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return []


@tool
def get_customer_profile(customer_id: str) -> Dict[str, Any]:
    """
    Retrieve customer profile information.
    
    Use this to get basic customer details like loyalty tier,
    contact information, and account status.
    
    Args:
        customer_id: Customer identifier
        
    Returns:
        Customer profile dictionary
    """
    try:
        collection = db_manager.get_collection(config.mongodb.customers_collection)
        
        customer = collection.find_one({"customer_id": customer_id})
        
        if customer:
            # Remove MongoDB _id field
            customer.pop("_id", None)
            logger.info(f"Retrieved profile for customer {customer_id}")
            return customer
        else:
            logger.warning(f"Customer {customer_id} not found")
            return {"error": "Customer not found"}
            
    except Exception as e:
        logger.error(f"Error retrieving customer profile: {e}")
        return {"error": str(e)}


@tool
def get_purchase_history(customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get customer's purchase history.
    
    Use this to see what the customer has bought before, helping
    you make better recommendations.
    
    Args:
        customer_id: Customer identifier
        limit: Maximum number of recent purchases to return
        
    Returns:
        List of purchase dictionaries
    """
    try:
        collection = db_manager.get_collection(config.mongodb.purchases_collection)
        
        purchases = list(collection.find(
            {"customer_id": customer_id}
        ).sort("date", -1).limit(limit))  # Most recent first
        
        # Format results
        results = []
        for purchase in purchases:
            purchase.pop("_id", None)  # Remove MongoDB _id
            results.append(purchase)
        
        logger.info(f"Retrieved {len(results)} purchases for customer {customer_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving purchase history: {e}")
        return []


# Note: Memory tools (search_memory_tool, manage_memory_tool) are created
# dynamically by LangMem and added to the agent in the graph definition

