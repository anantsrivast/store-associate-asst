from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
from src.database.mongodb_client import db_manager
from src.config import config
import logging

logger = logging.getLogger(__name__)


@tool
def search_products(
    query: str, 
    category: Optional[str] = None, 
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for products in the store catalog.
    
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
        ).sort("price", 1))
        
        # Format results
        results = []
        for product in products:
            results.append({
                "sku": product.get("sku"),
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price"),
                "description": product.get("description", "")[:200],
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
    Retrieve customer profile information including shoe size, preferences, and contact details.
    
    Use this to get structured customer data like:
    - Shoe size
    - Preferred brands
    - Loyalty tier
    - Contact information
    
    Args:
        customer_id: Customer identifier
        
    Returns:
        Customer profile dictionary with all structured data
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
def update_customer_profile(
    customer_id: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update customer profile with structured data like shoe size, preferences, etc.
    
    Use this when the customer shares factual information about themselves:
    - Shoe size: updates["shoe_size"] = 8
    - Preferred brands: updates["preferred_brands"] = ["Nike", "Adidas"]
    - Budget range: updates["budget_range"] = "$100-150"
    - Style preferences: updates["style_preferences"] = ["casual", "athletic"]
    
    Args:
        customer_id: Customer identifier
        updates: Dictionary of fields to update
        
    Returns:
        Success/failure message
    """
    try:
        collection = db_manager.get_collection(config.mongodb.customers_collection)
        
        # Update the customer document
        result = collection.update_one(
            {"customer_id": customer_id},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated customer {customer_id} with: {updates}")
            return {
                "success": True,
                "message": f"Successfully updated customer profile",
                "updated_fields": list(updates.keys())
            }
        else:
            logger.warning(f"No updates made for customer {customer_id}")
            return {
                "success": False,
                "message": "Customer not found or no changes made"
            }
            
    except Exception as e:
        logger.error(f"Error updating customer profile: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool
def get_purchase_history(
    customer_id: str, 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get customer's purchase history.
    
    Use this to see what the customer has bought before, helping
    you make better recommendations.
    
    Args:
        customer_id: Customer identifier
        limit: Maximum number of purchases to return
        
    Returns:
        List of purchase dictionaries
    """
    try:
        collection = db_manager.get_collection(config.mongodb.purchases_collection)
        
        purchases = list(collection.find(
            {"customer_id": customer_id}
        ).sort("purchase_date", -1).limit(limit))
        
        # Format results
        results = []
        for purchase in purchases:
            purchase.pop("_id", None)
            results.append(purchase)
        
        logger.info(f"Retrieved {len(results)} purchases for customer {customer_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving purchase history: {e}")
        return []