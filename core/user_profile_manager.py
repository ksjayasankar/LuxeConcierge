# luxury_leather_chatbot/core/user_profile_manager.py

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
from collections import defaultdict

# --- Absolute Imports ---
from config import BASE_DIR

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration ---
USER_PROFILES_DIR = os.path.join(BASE_DIR, "user_profiles")
os.makedirs(USER_PROFILES_DIR, exist_ok=True)

class UserProfileManager:
    """Manages user profiles for personalized recommendations."""
    
    def __init__(self):
        self.profiles_dir = USER_PROFILES_DIR
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the user profiles directory exists."""
        Path(self.profiles_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_profile_path(self, user_id: str) -> str:
        """Get the file path for a user's profile."""
        # Sanitize user_id to prevent directory traversal
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('_', '-'))
        return os.path.join(self.profiles_dir, f"{safe_user_id}.json")
    
    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user profile from disk, create new if doesn't exist."""
        profile_path = self._get_profile_path(user_id)
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                logger.info(f"Loaded profile for user {user_id}")
                return profile
            except Exception as e:
                logger.error(f"Error loading profile for {user_id}: {e}")
                # Return new profile if loading fails
        
        # Create new profile
        return self._create_new_profile(user_id)
    
    def _create_new_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a new user profile with default structure."""
        profile = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "interaction_count": 0,
            "preferences": {
                "styles": {},
                "colors": {},
                "price_range": {"min": None, "max": None, "avg": None, "all_prices": []},
                "materials": {},
                "occasions": {},
                "quality_scores": {
                    "avg_craftsmanship": None,
                    "avg_sustainability": None,
                    "all_craftsmanship": [],
                    "all_sustainability": []
                }
            },
            "interaction_history": [],
            "learned_insights": {
                "brand_affinity": None,
                "sustainability_importance": None,
                "quality_vs_price": None,
                "preferred_features": []
            }
        }
        logger.info(f"Created new profile for user {user_id}")
        return profile
    
    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        """Save user profile to disk."""
        profile_path = self._get_profile_path(user_id)
        
        try:
            # Update timestamp
            profile["last_updated"] = datetime.now().isoformat()
            
            # Write to temporary file first (atomic write)
            temp_path = f"{profile_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            
            # Rename to final path (atomic on most systems)
            os.replace(temp_path, profile_path)
            
            logger.debug(f"Saved profile for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile for {user_id}: {e}")
            return False
    
    def update_profile_with_interaction(
        self,
        user_id: str,
        user_message: str,
        extracted_criteria: Dict[str, Any],
        recommended_products: List[Dict[str, Any]],
        assistant_response: str
    ) -> Dict[str, Any]:
        """Update user profile with new interaction data."""
        profile = self.load_profile(user_id)
        
        # Increment interaction count
        profile["interaction_count"] += 1
        
        # Update preferences based on extracted criteria
        self._update_preferences(profile["preferences"], extracted_criteria, recommended_products)
        
        # Add to interaction history (keep last 50 interactions)
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": user_message,
            "extracted_criteria": extracted_criteria,
            "recommended_products": [p.get("name", "Unknown") for p in recommended_products],
            "assistant_response": assistant_response[:200] + "..." if len(assistant_response) > 200 else assistant_response
        }
        
        profile["interaction_history"].append(interaction)
        if len(profile["interaction_history"]) > 50:
            profile["interaction_history"] = profile["interaction_history"][-50:]
        
        # Update learned insights
        self._update_insights(profile)
        
        # Save updated profile
        self.save_profile(user_id, profile)
        
        return profile
    
    def _update_preferences(
        self, 
        preferences: Dict[str, Any], 
        criteria: Dict[str, Any],
        products: List[Dict[str, Any]]
    ):
        """Update preference counts and ranges based on interaction."""
        
        # Update style preferences
        if criteria.get("style"):
            style = criteria["style"]
            preferences["styles"][style] = preferences["styles"].get(style, 0) + 1
        
        # Update color preferences
        if criteria.get("color"):
            color = criteria["color"].lower()
            preferences["colors"][color] = preferences["colors"].get(color, 0) + 1
        
        # Update material preferences
        if criteria.get("material_preference"):
            material = criteria["material_preference"]
            preferences["materials"][material] = preferences["materials"].get(material, 0) + 1
        
        # Update occasion preferences
        if criteria.get("occasion"):
            occasion = criteria["occasion"]
            preferences["occasions"][occasion] = preferences["occasions"].get(occasion, 0) + 1
        
        # Update price range from criteria
        price_data = preferences["price_range"]
        if criteria.get("price_min") or criteria.get("price_max"):
            if criteria.get("price_min"):
                price_data["all_prices"].append(float(criteria["price_min"]))
            if criteria.get("price_max"):
                price_data["all_prices"].append(float(criteria["price_max"]))
        
        # Also learn from recommended products
        for product in products:
            # Track actual product prices user was shown
            if product.get("price"):
                price_data["all_prices"].append(float(product["price"]))
            
            # Track quality scores of products shown
            if product.get("craftsmanship_score"):
                preferences["quality_scores"]["all_craftsmanship"].append(
                    float(product["craftsmanship_score"])
                )
            if product.get("sustainability_score"):
                preferences["quality_scores"]["all_sustainability"].append(
                    float(product["sustainability_score"])
                )
        
        # Update aggregated price stats
        if price_data["all_prices"]:
            # Keep last 30 price points
            price_data["all_prices"] = price_data["all_prices"][-30:]
            price_data["avg"] = sum(price_data["all_prices"]) / len(price_data["all_prices"])
            price_data["min"] = min(price_data["all_prices"])
            price_data["max"] = max(price_data["all_prices"])
        
        # Update quality score averages
        quality = preferences["quality_scores"]
        if quality["all_craftsmanship"]:
            quality["all_craftsmanship"] = quality["all_craftsmanship"][-30:]
            quality["avg_craftsmanship"] = sum(quality["all_craftsmanship"]) / len(quality["all_craftsmanship"])
        
        if quality["all_sustainability"]:
            quality["all_sustainability"] = quality["all_sustainability"][-30:]
            quality["avg_sustainability"] = sum(quality["all_sustainability"]) / len(quality["all_sustainability"])
    
    def _update_insights(self, profile: Dict[str, Any]):
        """Derive insights from user interaction patterns."""
        insights = profile["learned_insights"]
        prefs = profile["preferences"]
        
        # Analyze style preferences
        if prefs["styles"]:
            top_styles = sorted(prefs["styles"].items(), key=lambda x: x[1], reverse=True)[:2]
            if top_styles:
                if "briefcase" in [s[0] for s in top_styles] or "messenger" in [s[0] for s in top_styles]:
                    insights["brand_affinity"] = "professional/business-oriented"
                elif "duffel" in [s[0] for s in top_styles] or "travel bag" in [s[0] for s in top_styles]:
                    insights["brand_affinity"] = "travel-focused"
                else:
                    insights["brand_affinity"] = "fashion-conscious"
        
        # Analyze sustainability importance
        if prefs["quality_scores"]["avg_sustainability"]:
            avg_sustain = prefs["quality_scores"]["avg_sustainability"]
            if avg_sustain >= 7.0:
                insights["sustainability_importance"] = "high"
            elif avg_sustain >= 6.0:
                insights["sustainability_importance"] = "moderate"
            else:
                insights["sustainability_importance"] = "low"
        
        # Analyze quality vs price preference
        if prefs["price_range"]["avg"] and prefs["quality_scores"]["avg_craftsmanship"]:
            avg_price = prefs["price_range"]["avg"]
            avg_craft = prefs["quality_scores"]["avg_craftsmanship"]
            
            if avg_craft >= 8.0 and avg_price >= 400:
                insights["quality_vs_price"] = "premium-quality-focused"
            elif avg_craft >= 7.0:
                insights["quality_vs_price"] = "balanced"
            else:
                insights["quality_vs_price"] = "budget-conscious"
        
        # Extract frequently mentioned features
        all_occasions = list(prefs["occasions"].keys())
        all_materials = list(prefs["materials"].keys())
        insights["preferred_features"] = all_occasions + all_materials
    
    def get_user_context_summary(self, user_id: str) -> str:
        """Generate a natural language summary of user preferences for the LLM."""
        profile = self.load_profile(user_id)
        
        if profile["interaction_count"] == 0:
            return "This is a new user with no previous interactions."
        
        prefs = profile["preferences"]
        insights = profile["learned_insights"]
        
        summary_parts = [f"User has had {profile['interaction_count']} previous interactions."]
        
        # Style preferences
        if prefs["styles"]:
            top_styles = sorted(prefs["styles"].items(), key=lambda x: x[1], reverse=True)[:2]
            if top_styles:
                summary_parts.append(
                    f"Prefers {top_styles[0][0]} (shown interest {top_styles[0][1]} times)"
                )
        
        # Color preferences
        if prefs["colors"]:
            top_colors = sorted(prefs["colors"].items(), key=lambda x: x[1], reverse=True)[:2]
            if top_colors:
                summary_parts.append(
                    f"Favors {', '.join([c[0] for c in top_colors])} colors"
                )
        
        # Price range
        if prefs["price_range"]["avg"]:
            summary_parts.append(
                f"Typical budget: ${prefs['price_range']['min']:.0f}-${prefs['price_range']['max']:.0f} "
                f"(avg: ${prefs['price_range']['avg']:.0f})"
            )
        
        # Material preferences
        if prefs["materials"]:
            top_material = max(prefs["materials"].items(), key=lambda x: x[1])
            summary_parts.append(f"Material preference: {top_material[0]}")
        
        # Quality preferences
        if prefs["quality_scores"]["avg_craftsmanship"]:
            summary_parts.append(
                f"Quality expectations: Craftsmanship ~{prefs['quality_scores']['avg_craftsmanship']:.1f}/10, "
                f"Sustainability ~{prefs['quality_scores']['avg_sustainability']:.1f}/10"
            )
        
        # Insights
        if insights["brand_affinity"]:
            summary_parts.append(f"Profile: {insights['brand_affinity']}")
        
        if insights["sustainability_importance"]:
            summary_parts.append(f"Sustainability importance: {insights['sustainability_importance']}")
        
        return " ".join(summary_parts)


# --- Singleton instance ---
_profile_manager = UserProfileManager()

def get_profile_manager() -> UserProfileManager:
    """Get the singleton UserProfileManager instance."""
    return _profile_manager