# luxury_leather_chatbot/api/chat_routes.py

import logging
from flask import Blueprint, request, jsonify, session
import asyncio

# --- Absolute Imports ---
try:
    from core.conversation_manager import process_user_message
    # *** Use the new readiness check ***
    from core.retrieval_module import is_retrieval_initialized
    from config import FALLBACK_ERROR_MESSAGE
except ImportError as e:
     print(f"Error importing in chat_routes: {e}")
     raise

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Blueprint ---
bp = Blueprint('chat_api', __name__)

# --- Chat Endpoint ---
@bp.route('/chat', methods=['POST'])
async def handle_chat():
    """Handles incoming user messages and returns the chatbot's response."""
    if not request.is_json:
        logger.warning("Received non-JSON request to /chat")
        return jsonify({"error": "Request must be JSON"}), 400

    # *** Add readiness check at the beginning (optional but good practice) ***
    if not is_retrieval_initialized():
        logger.warning("/chat request received but retrieval system is not ready.")
        # Provide a user-friendly message indicating temporary unavailability
        return jsonify({
            "response": "I'm currently getting ready to assist you. Please try again in a moment.",
            "products": []
            }), 503 # Service Unavailable

    data = request.get_json()
    user_message = data.get('message')

    if not user_message or not isinstance(user_message, str) or not user_message.strip():
        logger.warning("Received empty or invalid message in /chat request")
        return jsonify({"error": "Missing or invalid 'message' field"}), 400

    user_message = user_message.strip()

    # --- Session Management ---
    try:
        conversation_history = session.get('conversation_history', [])
        if not isinstance(conversation_history, list):
             logger.warning("Invalid conversation_history in session, resetting.")
             conversation_history = []
        # logger.debug(f"Loaded history (Session: {session}): {conversation_history}") # Debug session
    except Exception as e:
         logger.error(f"Error accessing session: {e}", exc_info=True)
         return jsonify({"error": "Session handling error."}), 500

    # --- Call Conversation Manager ---
    try:
        assistant_response_data = await process_user_message(conversation_history, user_message)

        # --- Update history ---
        conversation_history.append({"role": "user", "content": user_message})
        assistant_text = assistant_response_data.get("text_response", "")
        if not isinstance(assistant_text, str):
             logger.error("Received non-string text_response from conversation manager.")
             assistant_text = FALLBACK_ERROR_MESSAGE
        conversation_history.append({"role": "assistant", "content": assistant_text})

        # --- Limit history ---
        MAX_HISTORY_TURNS = 10
        MAX_HISTORY_LEN = MAX_HISTORY_TURNS * 2
        if len(conversation_history) > MAX_HISTORY_LEN:
            conversation_history = conversation_history[-MAX_HISTORY_LEN:]
            # logger.debug(f"History truncated to last {MAX_HISTORY_TURNS} turns.")

        session['conversation_history'] = conversation_history
        # logger.debug(f"Saved history: {conversation_history}") # Debug session save

        # --- Prepare response ---
        response_payload = {
            "response": assistant_text,
            "products": assistant_response_data.get("products", [])
        }
        if not isinstance(response_payload["products"], list):
             logger.error("Received non-list 'products' from conversation manager.")
             response_payload["products"] = []

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        return jsonify({"error": FALLBACK_ERROR_MESSAGE}), 500