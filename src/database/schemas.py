
from typing import Dict, Any, List
from pymongo import IndexModel, ASCENDING, DESCENDING


class MongoDBSchemas:
    """
    Defines all MongoDB collection schemas and indexes.
    
    This class provides schema definitions and index configurations
    for all collections used in the application.
    """
    
    @staticmethod
    def get_customer_schema() -> Dict[str, Any]:
        """
        Schema for the customers collection.
        
        Stores customer profile information including contact details,
        preferences, and loyalty status.
        """
        return {
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["customer_id", "name", "email"],
                    "properties": {
                        "customer_id": {
                            "bsonType": "string",
                            "description": "Unique customer identifier"
                        },
                        "name": {
                            "bsonType": "string",
                            "description": "Customer full name"
                        },
                        "email": {
                            "bsonType": "string",
                            "description": "Customer email address"
                        },
                        "phone": {
                            "bsonType": "string",
                            "description": "Customer phone number"
                        },
                        "loyalty_tier": {
                            "enum": ["bronze", "silver", "gold", "platinum"],
                            "description": "Customer loyalty tier"
                        },
                        "created_at": {
                            "bsonType": "date",
                            "description": "Account creation timestamp"
                        },
                        "updated_at": {
                            "bsonType": "date",
                            "description": "Last update timestamp"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def get_customer_indexes() -> List[IndexModel]:
        """
        Indexes for the customers collection.
        
        Returns:
            List of IndexModel objects for efficient queries
        """
        return [
            IndexModel([("customer_id", ASCENDING)], unique=True),
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("loyalty_tier", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)])
        ]
    
    @staticmethod
    def get_product_schema() -> Dict[str, Any]:
        """
        Schema for the products collection.
        
        Stores product catalog including details, pricing, and inventory.
        """
        return {
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["sku", "name", "category", "price"],
                    "properties": {
                        "sku": {
                            "bsonType": "string",
                            "description": "Stock Keeping Unit - unique product ID"
                        },
                        "name": {
                            "bsonType": "string",
                            "description": "Product name"
                        },
                        "category": {
                            "bsonType": "string",
                            "description": "Product category"
                        },
                        "brand": {
                            "bsonType": "string",
                            "description": "Product brand"
                        },
                        "price": {
                            "bsonType": "double",
                            "description": "Product price"
                        },
                        "description": {
                            "bsonType": "string",
                            "description": "Product description"
                        },
                        "variants": {
                            "bsonType": "array",
                            "description": "Available variants (sizes, colors)"
                        },
                        "inventory": {
                            "bsonType": "int",
                            "description": "Current inventory count"
                        },
                        "created_at": {
                            "bsonType": "date"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def get_product_indexes() -> List[IndexModel]:
        """Indexes for products collection"""
        return [
            IndexModel([("sku", ASCENDING)], unique=True),
            IndexModel([("category", ASCENDING)]),
            IndexModel([("brand", ASCENDING)]),
            IndexModel([("price", ASCENDING)]),
            # Text index for product search
            IndexModel([("name", "text"), ("description", "text")])
        ]
    
    @staticmethod
    def get_purchase_schema() -> Dict[str, Any]:
        """
        Schema for the purchases collection.
        
        Stores transaction history linking customers to products.
        """
        return {
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["order_id", "customer_id", "date", "items"],
                    "properties": {
                        "order_id": {
                            "bsonType": "string",
                            "description": "Unique order identifier"
                        },
                        "customer_id": {
                            "bsonType": "string",
                            "description": "Customer who made the purchase"
                        },
                        "date": {
                            "bsonType": "date",
                            "description": "Purchase date"
                        },
                        "items": {
                            "bsonType": "array",
                            "description": "List of purchased items",
                            "items": {
                                "bsonType": "object",
                                "required": ["sku", "quantity", "price"],
                                "properties": {
                                    "sku": {"bsonType": "string"},
                                    "quantity": {"bsonType": "int"},
                                    "price": {"bsonType": "double"}
                                }
                            }
                        },
                        "total_amount": {
                            "bsonType": "double",
                            "description": "Total purchase amount"
                        },
                        "store_location": {
                            "bsonType": "string",
                            "description": "Store where purchase was made"
                        },
                        "associate_id": {
                            "bsonType": "string",
                            "description": "Associate who helped with purchase"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def get_purchase_indexes() -> List[IndexModel]:
        """Indexes for purchases collection"""
        return [
            IndexModel([("order_id", ASCENDING)], unique=True),
            IndexModel([("customer_id", ASCENDING)]),
            IndexModel([("date", DESCENDING)]),
            IndexModel([("customer_id", ASCENDING), ("date", DESCENDING)])
        ]
    
    @staticmethod
    def get_memories_ttl_index() -> IndexModel:
        """
        TTL index for automatic memory expiration.
        
        This index automatically deletes documents after the TTL expires.
        MongoDB checks TTL indexes every 60 seconds.
        """
        from src.config import config
        return IndexModel(
            [("updated_at", ASCENDING)],
            expireAfterSeconds=config.memory.ttl_seconds
        )
