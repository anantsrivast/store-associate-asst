============================================================================
# FILE: src/data/synthetic_data.py
# DESCRIPTION: Generate realistic synthetic data for demonstrations
# ============================================================================

from faker import Faker
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)
fake = Faker()


class SyntheticDataGenerator:
    """
    Generates realistic synthetic data for the store assistant demo.
    
    This includes:
    - Customer profiles with varied demographics
    - Product catalog across multiple categories
    - Purchase history with realistic patterns
    - Initial memories for select customers
    """
    
    # Product categories and typical items
    CATEGORIES = {
        "athletic_shoes": {
            "brands": ["Nike", "Adidas", "Brooks", "Asics", "New Balance"],
            "price_range": (80, 180),
            "variants": ["sizes"]
        },
        "casual_shoes": {
            "brands": ["Nike", "Adidas", "Vans", "Converse", "Puma"],
            "price_range": (50, 120),
            "variants": ["sizes"]
        },
        "athletic_wear": {
            "brands": ["Nike", "Under Armour", "Lululemon", "Adidas"],
            "price_range": (30, 100),
            "variants": ["sizes", "colors"]
        },
        "casual_wear": {
            "brands": ["Nike", "Adidas", "Champion", "The North Face"],
            "price_range": (25, 80),
            "variants": ["sizes", "colors"]
        },
        "accessories": {
            "brands": ["Nike", "Adidas", "Under Armour"],
            "price_range": (15, 50),
            "variants": ["colors"]
        }
    }
    
    SIZES = {
        "shoes": ["6", "6.5", "7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"],
        "clothing": ["XS", "S", "M", "L", "XL", "XXL"]
    }
    
    COLORS = ["Black", "White", "Navy", "Gray", "Red", "Blue", "Green"]
    
    def __init__(self, num_customers: int = 50, num_products: int = 200):
        """
        Initialize the generator.
        
        Args:
            num_customers: Number of customer profiles to generate
            num_products: Number of products to generate
        """
        self.num_customers = num_customers
        self.num_products = num_products
        Faker.seed(42)  # For reproducibility
        random.seed(42)
    
    def generate_customers(self) -> List[Dict[str, Any]]:
        """
        Generate customer profiles.
        
        Returns:
            List of customer dictionaries
        """
        logger.info(f"Generating {self.num_customers} customer profiles")
        
        customers = []
        loyalty_tiers = ["bronze", "silver", "gold", "platinum"]
        
        for i in range(self.num_customers):
            customer_id = f"customer_{i+1:04d}"
            
            # Generate realistic profile
            customer = {
                "customer_id": customer_id,
                "name": fake.name(),
                "email": fake.email(),
                "phone": fake.phone_number(),
                "loyalty_tier": random.choices(
                    loyalty_tiers,
                    weights=[40, 30, 20, 10]  # More bronze, fewer platinum
                )[0],
                "created_at": fake.date_time_between(
                    start_date="-2y",
                    end_date="now"
                ),
                "updated_at": datetime.now()
            }
            
            customers.append(customer)
        
        logger.info(f"Generated {len(customers)} customers")
        return customers
    
    def generate_products(self) -> List[Dict[str, Any]]:
        """
        Generate product catalog.
        
        Returns:
            List of product dictionaries
        """
        logger.info(f"Generating {self.num_products} products")
        
        products = []
        product_names = {
            "athletic_shoes": [
                "Pegasus Runner", "Air Zoom", "UltraBoost", "GEL-Nimbus",
                "Fresh Foam", "Ghost Runner", "Speedgoat"
            ],
            "casual_shoes": [
                "Classic Sneaker", "Low Top", "High Top", "Slip-On",
                "Court Vision", "Stan Smith"
            ],
            "athletic_wear": [
                "Training Shirt", "Running Tank", "Training Shorts",
                "Running Tights", "Sports Bra", "Training Jacket"
            ],
            "casual_wear": [
                "Hoodie", "T-Shirt", "Sweatpants", "Joggers",
                "Zip-Up", "Crew Neck"
            ],
            "accessories": [
                "Sports Socks", "Headband", "Wristband", "Water Bottle",
                "Gym Bag", "Cap"
            ]
        }
        
        sku_counter = 1
        
        for category, config in self.CATEGORIES.items():
            names = product_names[category]
            brands = config["brands"]
            price_min, price_max = config["price_range"]
            
            # Generate multiple products per category
            products_per_category = self.num_products // len(self.CATEGORIES)
            
            for _ in range(products_per_category):
                name = random.choice(names)
                brand = random.choice(brands)
                
                # Generate variants
                variants = []
                if "sizes" in config["variants"]:
                    size_type = "shoes" if "shoes" in category else "clothing"
                    variants.extend([{"type": "size", "value": s} for s in self.SIZES[size_type]])
                
                if "colors" in config["variants"]:
                    variants.extend([{"type": "color", "value": c} for c in random.sample(self.COLORS, 3)])
                
                product = {
                    "sku": f"SKU{sku_counter:06d}",
                    "name": f"{brand} {name}",
                    "category": category,
                    "brand": brand,
                    "price": round(random.uniform(price_min, price_max), 2),
                    "description": self._generate_product_description(category, brand, name),
                    "variants": variants,
                    "inventory": random.randint(0, 100),
                    "created_at": datetime.now()
                }
                
                products.append(product)
                sku_counter += 1
        
        logger.info(f"Generated {len(products)} products")
        return products
    
    def _generate_product_description(self, category: str, brand: str, name: str) -> str:
        """Generate realistic product description"""
        descriptions = {
            "athletic_shoes": f"High-performance running shoe designed for comfort and speed. Features responsive cushioning and breathable mesh upper.",
            "casual_shoes": f"Classic everyday sneaker with timeless style. Comfortable fit perfect for all-day wear.",
            "athletic_wear": f"Performance {category.split('_')[1]} designed for intense workouts. Moisture-wicking fabric keeps you dry.",
            "casual_wear": f"Comfortable {category.split('_')[1]} perfect for everyday wear. Soft fabric with modern fit.",
            "accessories": f"Essential training accessory for athletes. Durable construction built to last."
        }
        
        return descriptions.get(category, "Quality product from {brand}.")
    
    def generate_purchases(self, customers: List[Dict], products: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate purchase history for customers.
        
        Creates realistic purchase patterns with:
        - Seasonal variations
        - Brand loyalty
        - Category preferences
        
        Args:
            customers: List of customer profiles
            products: List of products
            
        Returns:
            List of purchase dictionaries
        """
        logger.info("Generating purchase history")
        
        purchases = []
        order_counter = 1
        
        # Generate 3-10 purchases per customer
        for customer in customers[:30]:  # Only for first 30 customers to keep it manageable
            num_purchases = random.randint(3, 10)
            
            # Pick favorite brands for this customer
            all_brands = list(set(p["brand"] for p in products))
            favorite_brands = random.sample(all_brands, min(2, len(all_brands)))
            
            for _ in range(num_purchases):
                # Generate purchase date (last 12 months)
                purchase_date = fake.date_time_between(
                    start_date="-1y",
                    end_date="now"
                )
                
                # Select 1-3 items
                num_items = random.randint(1, 3)
                
                # Prefer favorite brands (70% of time)
                if random.random() < 0.7:
                    available_products = [p for p in products if p["brand"] in favorite_brands]
                else:
                    available_products = products
                
                items = []
                total_amount = 0
                
                for _ in range(num_items):
                    product = random.choice(available_products)
                    quantity = 1
                    price = product["price"]
                    
                    items.append({
                        "sku": product["sku"],
                        "name": product["name"],
                        "quantity": quantity,
                        "price": price
                    })
                    
                    total_amount += price * quantity
                
                purchase = {
                    "order_id": f"ORDER{order_counter:08d}",
                    "customer_id": customer["customer_id"],
                    "date": purchase_date,
                    "items": items,
                    "total_amount": round(total_amount, 2),
                    "store_location": random.choice(["Main St", "Downtown", "Mall", "Outlet"]),
                    "associate_id": f"ASSOC{random.randint(1, 10):03d}"
                }
                
                purchases.append(purchase)
                order_counter += 1
        
        logger.info(f"Generated {len(purchases)} purchases")
        return purchases
    
    def generate_initial_memories(self, customers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Generate initial memories for demo customers.
        
        Creates a few pre-populated memories to make demos more interesting.
        
        Args:
            customers: List of customer profiles
            
        Returns:
            Dictionary mapping customer_id to list of memories
        """
        logger.info("Generating initial memories for demo customers")
        
        memories = {}
        
        # Create memories for first 5 customers
        for customer in customers[:5]:
            customer_id = customer["customer_id"]
            customer_memories = []
            
            # Add some preferences
            preferences = [
                {
                    "namespace": ("customers", customer_id, "preferences"),
                    "key": "shoe_size",
                    "value": {
                        "preference_type": "shoe_size",
                        "value": random.choice(self.SIZES["shoes"]),
                        "confidence": 1.0,
                        "source": "conversation",
                        "first_observed": datetime.now().isoformat(),
                        "last_confirmed": datetime.now().isoformat(),
                        "times_observed": 1
                    }
                },
                {
                    "namespace": ("customers", customer_id, "preferences"),
                    "key": "favorite_brand",
                    "value": {
                        "preference_type": "favorite_brand",
                        "value": random.choice(["Nike", "Adidas", "Brooks"]),
                        "confidence": 0.9,
                        "source": "conversation",
                        "first_observed": datetime.now().isoformat(),
                        "last_confirmed": datetime.now().isoformat(),
                        "times_observed": 2
                    }
                }
            ]
            
            customer_memories.extend(preferences)
            
            # Add an episode
            episode = {
                "namespace": ("customers", customer_id, "episodes"),
                "key": f"episode_{datetime.now().isoformat()}",
                "value": {
                    "date": datetime.now().isoformat(),
                    "summary": f"Customer visited looking for running shoes. Discussed various options and made a purchase.",
                    "customer_needs": ["running shoes", "athletic wear"],
                    "products_discussed": ["Nike Pegasus", "Adidas UltraBoost"],
                    "outcome": "Purchase completed",
                    "key_insights": "Customer is training for a marathon",
                    "sentiment": "positive",
                    "duration_minutes": 15,
                    "associate_id": "ASSOC001",
                    "created_at": datetime.now().isoformat()
                }
            }
            
            customer_memories.append(episode)
            
            memories[customer_id] = customer_memories
        
        logger.info(f"Generated initial memories for {len(memories)} customers")
        return memories
