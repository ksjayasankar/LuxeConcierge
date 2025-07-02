# luxury_leather_chatbot/api/chat_routes.py

import logging
from flask import Blueprint, request, jsonify, session
import asyncio
import uuid

# --- Absolute Imports ---
try:
    from core.conversation_manager import process_user_message
    from core.retrieval_module import is_retrieval_initialized
    from core.user_profile_manager import get_profile_manager
    from config import FALLBACK_ERROR_MESSAGE
except ImportError as e:
     print(f"Error importing in chat_routes: {e}")
     raise

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Blueprint ---
bp = Blueprint('chat_api', __name__)

# --- Helper to get/create user ID ---
def get_or_create_user_id():
    """Get user ID from session or create a new one."""
    if 'user_id' not in session:
        # Generate a unique user ID
        session['user_id'] = f"user_{uuid.uuid4().hex[:12]}"
        session.permanent = True
        logger.info(f"Created new user ID: {session['user_id']}")
    return session['user_id']

# --- Chat Endpoint ---
@bp.route('/chat', methods=['POST'])
async def handle_chat():
    """Handles incoming user messages and returns the chatbot's response."""
    if not request.is_json:
        logger.warning("Received non-JSON request to /chat")
        return jsonify({"error": "Request must be JSON"}), 400

    # Check if retrieval system is ready
    if not is_retrieval_initialized():
        logger.warning("/chat request received but retrieval system is not ready.")
        return jsonify({
            "response": "I'm currently getting ready to assist you. Please try again in a moment.",
            "products": []
            }), 503

    data = request.get_json()
    user_message = data.get('message')

    if not user_message or not isinstance(user_message, str) or not user_message.strip():
        logger.warning("Received empty or invalid message in /chat request")
        return jsonify({"error": "Missing or invalid 'message' field"}), 400

    user_message = user_message.strip()

    # --- Get or create user ID ---
    user_id = get_or_create_user_id()
    
    # --- Load user profile ---
    profile_manager = get_profile_manager()
    user_profile = profile_manager.load_profile(user_id)
    
    # --- Generate user context summary for the LLM ---
    user_context = profile_manager.get_user_context_summary(user_id)
    
    # --- Session Management ---
    try:
        conversation_history = session.get('conversation_history', [])
        if not isinstance(conversation_history, list):
             logger.warning("Invalid conversation_history in session, resetting.")
             conversation_history = []
    except Exception as e:
         logger.error(f"Error accessing session: {e}", exc_info=True)
         return jsonify({"error": "Session handling error."}), 500

    # --- Call Conversation Manager with user context ---
    try:
        assistant_response_data = await process_user_message(
            conversation_history, 
            user_message,
            user_context=user_context,  # Pass user context
            user_profile=user_profile    # Pass full profile for advanced use
        )

        # --- Extract response components ---
        assistant_text = assistant_response_data.get("text_response", "")
        recommended_products = assistant_response_data.get("products", [])
        extracted_criteria = assistant_response_data.get("extracted_criteria", {})
        
        if not isinstance(assistant_text, str):
             logger.error("Received non-string text_response from conversation manager.")
             assistant_text = FALLBACK_ERROR_MESSAGE

        # --- Update user profile with this interaction ---
        profile_manager.update_profile_with_interaction(
            user_id=user_id,
            user_message=user_message,
            extracted_criteria=extracted_criteria,
            recommended_products=recommended_products,
            assistant_response=assistant_text
        )
        
        # --- Update conversation history ---
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": assistant_text})

        # --- Limit history ---
        MAX_HISTORY_TURNS = 10
        MAX_HISTORY_LEN = MAX_HISTORY_TURNS * 2
        if len(conversation_history) > MAX_HISTORY_LEN:
            conversation_history = conversation_history[-MAX_HISTORY_LEN:]

        session['conversation_history'] = conversation_history

        # --- Prepare response ---
        response_payload = {
            "response": assistant_text,
            "products": recommended_products
        }
        
        # Add debug info in development
        if logger.isEnabledFor(logging.DEBUG):
            response_payload["_debug"] = {
                "user_id": user_id,
                "interaction_count": user_profile["interaction_count"] + 1,
                "extracted_criteria": extracted_criteria
            }

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        return jsonify({"error": FALLBACK_ERROR_MESSAGE}), 500

# --- Profile endpoint (optional) ---
@bp.route('/profile', methods=['GET'])
def get_user_profile():
    """Get current user's profile summary."""
    try:
        user_id = get_or_create_user_id()
        profile_manager = get_profile_manager()
        profile = profile_manager.load_profile(user_id)
        
        # Return a summary, not the full profile (for privacy)
        summary = {
            "user_id": user_id,
            "interaction_count": profile["interaction_count"],
            "preferences_summary": profile_manager.get_user_context_summary(user_id),
            "top_styles": sorted(
                profile["preferences"]["styles"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3] if profile["preferences"]["styles"] else [],
            "price_range": {
                "min": profile["preferences"]["price_range"]["min"],
                "max": profile["preferences"]["price_range"]["max"]
            } if profile["preferences"]["price_range"]["min"] else None
        }
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve profile"}), 500