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
) -> str:
    """
    Search for products in the store catalog using Atlas Full-Text Search.
    
    Use this tool when the customer asks about products or you need
    to find product information.
    
    Args:
        query: Search query (product name, keywords, description, brand)
        category: Optional category filter (ignored for now - searches all)
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string with product details
    """
    try:
        collection = db_manager.get_collection(config.mongodb.products_collection)
        
        # Simple Atlas Search - search everything!
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "text": {
                        "query": query,
                        "path": {"wildcard": "*"},  # Search ALL text fields
                        "fuzzy": {
                            "maxEdits": 1
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "search_score": {"$meta": "searchScore"}
                }
            },
            {
                "$limit": max_results
            }
        ]
        
        logger.info(f"Atlas Search for '{query}' (category hint: {category})")
        
        # Execute search
        products = list(collection.aggregate(pipeline))
        
        if not products:
            return f"No products found matching '{query}'"
        
        # Format results as a readable string
        result_text = f"Found {len(products)} products:\n\n"
        
        for i, product in enumerate(products, 1):
            result_text += f"{i}. **{product.get('name', 'Unknown')}**\n"
            result_text += f"   • Brand: {product.get('brand', 'N/A')}\n"
            result_text += f"   • Price: ${product.get('price', 'N/A'):.2f}\n"
            result_text += f"   • Category: {product.get('category', 'N/A')}\n"
            result_text += f"   • SKU: {product.get('sku', 'N/A')}\n"
            result_text += f"   • Relevance: {product.get('search_score', 0):.2f}\n"
            
            # Add description if available
            description = product.get('description', '')
            if description:
                desc_preview = description[:150] + "..." if len(description) > 150 else description
                result_text += f"   • Description: {desc_preview}\n"
            
            # Add variants if available
            variants = product.get('variants', [])
            if variants:
                sizes = [v.get('size') for v in variants if v.get('size')]
                if sizes:
                    result_text += f"   • Available sizes: {', '.join(map(str, sizes))}\n"
            
            result_text += "\n"
        
        logger.info(f"Atlas Search returned {len(products)} results")
        return result_text
        
    except Exception as e:
        logger.error(f"Error searching products: {e}", exc_info=True)
        return f"Sorry, I encountered an error searching for products: {str(e)}"
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