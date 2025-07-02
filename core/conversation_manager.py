# luxury_leather_chatbot/core/conversation_manager.py

import logging
import json
import random
import copy
from typing import List, Dict, Any, Optional

# --- Absolute Imports ---
from core.llm_interface import generate_text
from core.retrieval_module import (
    get_product_candidates_hybrid,
    is_retrieval_initialized,
    _product_data_store
)
from core.reranking_module import reflect_and_rerank
from config import (
    CONVERSATIONAL_MODEL_NAME, FALLBACK_ERROR_MESSAGE, RERANKING_TOP_N,
    RERANKING_MODEL_NAME
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Helper Functions (Prompts & Parsing) ---

def _construct_conversation_prompt(
    conversation_history: List[Dict[str, str]], 
    user_message: str,
    user_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Constructs the prompt for the conversational LLM with user context.
    """
    # Base system prompt
    base_system_prompt = """You are a helpful and knowledgeable personal shopper for a luxury leather brand.
Your goal is to understand the user's needs through conversation and assist them effectively.

Analyze the conversation history and the latest user message. Determine the user's primary intent."""

    # Add user context if available
    if user_context and user_context != "This is a new user with no previous interactions.":
        user_context_section = f"""

**User Profile Context:**
{user_context}

Consider this profile information when making recommendations, but also be open to the user exploring new preferences. 
If the current request conflicts with past preferences, prioritize the current request but you may acknowledge the difference 
(e.g., "I see you usually prefer darker colors, but let me show you some options in navy...").
"""
    else:
        user_context_section = ""

    # Rest of the prompt
    action_section = """

**Possible Intents & Actions:**

1.  **Find Products (Action: `TRIGGER_RETRIEVAL`)**:
    *   If the user is asking for recommendations or searching for products with specific features.
    *   **Requires *reasonably specific* criteria.** Ideally, this includes style AND color, occasion, price range, or specific features. Do NOT trigger retrieval if only a broad style (like 'briefcase', 'bag') is mentioned without further context or details from the user message or recent history.
    *   Extract key criteria (style, color, material, occasion, price, quality scores) using the specified format below.
    *   Formulate a concise `retrieval_query` summarizing the core need for semantic search.
    *   When extracting criteria, consider both the current message AND the user's profile preferences as defaults when not explicitly stated.

2.  **Compare Specific Products (Action: `COMPARE_PRODUCTS`)**:
    *   If the user explicitly asks to compare specific products mentioned recently in the conversation (e.g., "compare X and Y", "what's the difference between A, B, C?", "tell me more about X vs Y").
    *   Extract the **exact names** of the products mentioned for comparison into the `products_to_compare` list. ONLY include names that seem to be actual product names from the brand.

3.  **Ask for Clarification (Action: `ASK_CLARIFICATION`)**:
    *   **Trigger this if the user provides only general criteria (like just the style or a vague use case) and more information would significantly help narrow down choices.** For example, if they just say "I want a briefcase", ask about color, material, or intended use.
    *   Also trigger if crucial information for search is missing (e.g., no style or context provided at all) or criteria are ambiguous/conflicting.
    *   Ask ONE targeted question to get the most helpful next piece of information (e.g., "Okay, a briefcase! What color are you thinking of?" or "Got it. Are you looking for a briefcase primarily for work, travel, or something else?").
    *   If the user has established preferences in their profile, you might reference them in your clarification (e.g., "I see you usually go for dark brown leather. Are you looking for something similar this time?").

4.  **Answer General Question (Action: `ANSWER_QUESTION`)**:
    *   If the user asked a general question about the brand, materials, care, policy, etc. (that isn't a specific product comparison). Answer concisely based on general knowledge.

5.  **Casual Chat (Action: `CHIT_CHAT`)**:
    *   If the user message is conversational small talk. Respond politely.

**Extraction Criteria Format (for `TRIGGER_RETRIEVAL`):**
*   `style`: MUST be one of: briefcase, messenger, tote, duffel, handbag, slim bag, sling, crossbody, travel bag, trolley (string, optional)
*   `color`: e.g., "dark brown" (string, optional)
*   `occasion`: e.g., "work" (string, optional)
*   `price_min`, `price_max`: (number, optional)
*   `material_preference`: "real_leather", "vegan", "synthetic", "any" (string, optional, default 'any') - Infer based on user terms (full-grain -> real_leather, faux -> vegan/synthetic).
*   `min_craftsmanship`, `min_sustainability`: (number, optional) - Set based on quality indicators (e.g., high quality -> 8.0, sustainable -> 7.0).

**Output Format:**
Return ONLY a single JSON object with the following structure:
{{
  "action": "TRIGGER_RETRIEVAL | COMPARE_PRODUCTS | ASK_CLARIFICATION | ANSWER_QUESTION | CHIT_CHAT",
  "extracted_criteria": {{ ... }},       // REQUIRED for TRIGGER_RETRIEVAL (use format above). Include ONLY criteria explicitly mentioned or strongly implied. Omit keys if no info.
  "retrieval_query": "...",           // REQUIRED for TRIGGER_RETRIEVAL (Concise NL summary for semantic search).
  "products_to_compare": ["...", "..."], // REQUIRED for COMPARE_PRODUCTS (List of exact product names).
  "clarification_question": "...",    // REQUIRED for ASK_CLARIFICATION.
  "answer_text": "...",               // REQUIRED for ANSWER_QUESTION or CHIT_CHAT.
  "confidence_score": 0.0-1.0         // Optional: Your confidence.
}}

**Instructions:**
- Prioritize `COMPARE_PRODUCTS` if the user clearly asks to compare specific items mentioned recently. Extract their names accurately.
- If searching, map user style descriptions to the allowed `style` values. Extract other criteria carefully.
- **If only a broad style is provided without other details in the current message or recent context, prefer `ASK_CLARIFICATION` over `TRIGGER_RETRIEVAL` unless the conversation history strongly suggests the user wants a general overview.**
- When user preferences from their profile could help fill in missing criteria, use them as smart defaults but don't override explicit current requests.
- Ensure the output is always valid JSON. Do not add explanations outside the JSON object.
"""

    # Combine all parts
    system_prompt = base_system_prompt + user_context_section + action_section
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Limit history passed to this initial prompt if it gets too long
    MAX_HISTORY_FOR_ACTION = 6
    messages.extend(conversation_history[-(MAX_HISTORY_FOR_ACTION*2):])
    messages.append({"role": "user", "content": user_message})

    return messages


def _parse_llm_action_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response to extract action data."""
    if not response_text: 
        logger.error("Received empty response from conversational LLM.")
        return None
    
    try:
        # Find JSON block robustly
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        if json_start == -1 or json_end == -1 or json_end < json_start:
            logger.error(f"Could not find valid JSON object delimiters in LLM action response: {response_text}")
            try: 
                action_data = json.loads(response_text.strip())
            except json.JSONDecodeError: 
                logger.error(f"Direct JSON loading failed for response: {response_text}")
                return None
        else: 
            json_str = response_text[json_start:json_end+1]
            action_data = json.loads(json_str)

        logger.debug(f"Parsed LLM action data: {action_data}")

        # Basic validation
        if "action" not in action_data: 
            logger.error(f"LLM action missing 'action' key: {action_data}")
            return None
        
        action = action_data["action"]

        # Action-specific validation
        if action == "TRIGGER_RETRIEVAL":
            if "extracted_criteria" not in action_data: 
                action_data["extracted_criteria"] = {}
                logger.warning("LLM action TRIGGER_RETRIEVAL missing 'extracted_criteria', adding empty dict.")
            if not isinstance(action_data["extracted_criteria"], dict): 
                logger.error(f"'extracted_criteria' not a dict: {action_data['extracted_criteria']}")
                action_data["extracted_criteria"] = {}
            if not action_data.get("retrieval_query"): 
                logger.error("Action TRIGGER_RETRIEVAL missing 'retrieval_query'.")
                return None
            # Optional: Validate style within criteria
            if action_data.get("extracted_criteria", {}).get("style"):
                allowed_styles = {"briefcase", "messenger", "tote", "duffel", "handbag", "slim bag", "sling", "crossbody", "travel bag", "trolley"}
                extracted_style = action_data["extracted_criteria"]["style"].lower()
                if extracted_style not in allowed_styles: 
                    logger.warning(f"LLM extracted an invalid style: '{extracted_style}'.")

        elif action == "COMPARE_PRODUCTS":
            if "products_to_compare" not in action_data: 
                logger.error("Action COMPARE_PRODUCTS missing 'products_to_compare'.")
                return None
            if not isinstance(action_data.get("products_to_compare"), list): 
                logger.error("Action COMPARE_PRODUCTS: 'products_to_compare' is not a list.")
                return None
            if not all(isinstance(name, str) for name in action_data["products_to_compare"]): 
                logger.error("Action COMPARE_PRODUCTS: 'products_to_compare' contains non-string elements.")
                return None
            if len(action_data["products_to_compare"]) < 2: 
                logger.warning("Action COMPARE_PRODUCTS: Fewer than 2 products listed for comparison.")

        elif action == "ASK_CLARIFICATION":
            if not action_data.get("clarification_question"): 
                logger.error("Action ASK_CLARIFICATION missing 'clarification_question'.")
                return None

        elif action == "ANSWER_QUESTION" or action == "CHIT_CHAT":
            if not action_data.get("answer_text"): 
                logger.error(f"Action {action} missing 'answer_text'.")
                return None

        # Ensure extracted_criteria exists even if not TRIGGER_RETRIEVAL for safety
        if "extracted_criteria" not in action_data: 
            action_data["extracted_criteria"] = {}

        return action_data
        
    except json.JSONDecodeError as e: 
        logger.error(f"Failed decode LLM action JSON: {e}", exc_info=True)
        logger.debug(f"Problematic text: {response_text}")
        return None
    except Exception as e: 
        logger.error(f"Unexpected error parsing LLM action response: {e}", exc_info=True)
        return None


def _enhance_criteria_with_profile(
    extracted_criteria: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Enhance extracted criteria with user profile preferences as defaults.
    Only fills in missing values, doesn't override explicit criteria.
    """
    if not user_profile or user_profile.get("interaction_count", 0) < 3:
        # Don't use profile until we have enough data
        return extracted_criteria
    
    enhanced = extracted_criteria.copy()
    prefs = user_profile.get("preferences", {})
    
    # Add default price range if not specified
    if not enhanced.get("price_min") and not enhanced.get("price_max"):
        price_data = prefs.get("price_range", {})
        if price_data.get("min") and price_data.get("max"):
            # Use a slightly wider range than historical
            buffer = 0.15  # 15% buffer
            avg_price = price_data.get("avg", 0)
            if avg_price:
                enhanced["price_min"] = max(50, price_data["min"] * (1 - buffer))
                enhanced["price_max"] = price_data["max"] * (1 + buffer)
                logger.debug(f"Added price range from profile: ${enhanced['price_min']:.0f}-${enhanced['price_max']:.0f}")
    
    # Add quality preferences if not specified
    quality_scores = prefs.get("quality_scores", {})
    if not enhanced.get("min_craftsmanship") and quality_scores.get("avg_craftsmanship"):
        # Set slightly below their average to allow variety
        enhanced["min_craftsmanship"] = max(6.0, quality_scores["avg_craftsmanship"] - 1.0)
        logger.debug(f"Added min craftsmanship from profile: {enhanced['min_craftsmanship']:.1f}")
    
    if not enhanced.get("min_sustainability") and quality_scores.get("avg_sustainability"):
        if quality_scores["avg_sustainability"] >= 7.0:  # User cares about sustainability
            enhanced["min_sustainability"] = max(6.0, quality_scores["avg_sustainability"] - 1.0)
            logger.debug(f"Added min sustainability from profile: {enhanced['min_sustainability']:.1f}")
    
    return enhanced


def _build_comparison_prompt(products_data: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
    """Builds the prompt for the LLM to generate a factual comparison."""
    prompt_intro = f"""You are an assistant comparing products based ONLY on the provided information.
Do not add outside knowledge or invent features. If information is missing for a specific product or aspect, state that clearly (e.g., "not specified").

The user asked: "{user_query}"

Compare the following products based on material, price, durability indicators (like craftsmanship score), colors, and notable features mentioned in their descriptions. Summarize the key differences relevant to the user's query.
"""

    product_details_text = ""
    for i, p_data in enumerate(products_data):
        # Include key fields for comparison
        product_details_text += f"\n--- Product {i+1}: {p_data.get('name', 'Unknown Name')} ---\n"
        product_details_text += f"Price: ${p_data.get('price', 'N/A'):.2f}\n"
        product_details_text += f"Style: {p_data.get('style', 'Not specified')}\n"
        product_details_text += f"Material (leather_type): {p_data.get('leather_type', 'Not specified')}\n"
        product_details_text += f"Color(s): {p_data.get('color', 'Not specified')}\n"
        product_details_text += f"Craftsmanship Score: {p_data.get('craftsmanship_score', 'N/A')}\n"
        product_details_text += f"Sustainability Score: {p_data.get('sustainability_score', 'N/A')}\n"
        product_details_text += f"Occasion(s): {p_data.get('occasion', 'Not specified')}\n"
        product_details_text += f"Description: {p_data.get('description', 'No description provided.')}\n"
        product_details_text += "---\n"

    full_prompt = prompt_intro + product_details_text + "\nComparison Task: Generate the comparison now."

    return [{"role": "user", "content": full_prompt}]


def _check_profile_alignment(products: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> bool:
    """Check if recommended products align with user's typical preferences."""
    if not products or not user_profile:
        return True
    
    prefs = user_profile.get("preferences", {})
    
    # Check if products match preferred styles
    if prefs.get("styles"):
        top_style = max(prefs["styles"].items(), key=lambda x: x[1])[0]
        matching_styles = sum(1 for p in products if p.get("style") == top_style)
        if matching_styles / len(products) >= 0.5:
            return True
    
    # Check price alignment
    if prefs.get("price_range", {}).get("avg"):
        avg_profile_price = prefs["price_range"]["avg"]
        avg_product_price = sum(p.get("price", 0) for p in products) / len(products)
        if abs(avg_product_price - avg_profile_price) / avg_profile_price < 0.3:  # Within 30%
            return True
    
    return False


def _generate_personalized_response(
    response_type: str,
    products: List[Dict[str, Any]],
    user_profile: Optional[Dict[str, Any]],
    relaxation_info: Optional[str] = None,
    mismatched_criteria: List[str] = None
) -> str:
    """
    Generate responses that acknowledge user history when relevant.
    """
    # For new users or users with few interactions, use the standard responses
    if not user_profile or user_profile.get("interaction_count", 0) < 3:
        # Use existing response generation logic
        return _generate_standard_response(response_type, products, relaxation_info, mismatched_criteria)
    
    # For returning users, add personalized touches
    insights = user_profile.get("learned_insights", {})
    prefs = user_profile.get("preferences", {})
    
    if response_type == "recommendations":
        num_products = len(products)
        product_names = [p.get('name', 'Unknown') for p in products]
        
        # Check if recommendations align with profile
        profile_aligned = _check_profile_alignment(products, user_profile)
        
        if profile_aligned and num_products > 0:
            # Recommendations match their usual preferences
            intro_phrases = [
                "Based on your preferences, I found",
                "These match your usual style:",
                "I think you'll like these options:",
                "Here are some pieces that fit your taste:"
            ]
        else:
            # Recommendations are different from usual
            intro_phrases = [
                "I found something a bit different this time:",
                "These are outside your usual picks, but might interest you:",
                "Exploring some new options for you:",
                "Here's something fresh to consider:"
            ]
        
        intro = random.choice(intro_phrases)
        
        if num_products == 1:
            return f"{intro} the '{product_names[0]}'. Does this work for what you have in mind?"
        else:
            product_list = ", ".join([f"'{name}'" for name in product_names[:-1]]) + f", and '{product_names[-1]}'"
            return f"{intro} {product_list}. Any of these catching your eye?"
    
    # For other response types, fall back to standard
    return _generate_standard_response(response_type, products, relaxation_info, mismatched_criteria)


def _generate_standard_response(
    response_type: str,
    products: List[Dict[str, Any]],
    relaxation_info: Optional[str] = None,
    mismatched_criteria: List[str] = None
) -> str:
    """Original response generation logic (fallback for non-personalized responses)."""
    if response_type == "recommendations":
        num_products = len(products)
        product_names = [p.get('name', 'Unknown') for p in products]
        
        # Build preamble based on relaxation/mismatch
        preamble = ""
        if mismatched_criteria:
            mismatch_details = " and ".join(mismatched_criteria)
            preamble_options = [
                f"Okay, I couldn't find anything available right now {mismatch_details}, but based on your other preferences, these might be good alternatives: ",
                f"While I don't have an exact match {mismatch_details}, these options are quite relevant otherwise and might interest you: ",
                f"Hmm, nothing matched perfectly {mismatch_details}. However, take a look at these suggestions: "
            ]
            preamble = random.choice(preamble_options)
        elif relaxation_info:
            relaxation_preamble_options = [
                f"Okay, I couldn't find an exact match for *all* your preferences, but by slightly relaxing the **{relaxation_info}**, ",
                f"While nothing matched perfectly, adjusting the **{relaxation_info}** allowed me to find these options: ",
                f"To find some relevant choices, I relaxed the **{relaxation_info}** a bit. "
            ]
            preamble = random.choice(relaxation_preamble_options)
        
        # Build recommendation text
        if num_products == 1:
            product_list_str = f"the '{product_names[0]}'"
            recommendation_text = f"I found this option you might like: {product_list_str}."
            follow_up = "Does this seem like a good fit?"
        else:
            product_list_str = ", ".join([f"'{name}'" for name in product_names[:-1]]) + f", and '{product_names[-1]}'"
            recommendation_text = f"I found these {num_products} options you might like: {product_list_str}."
            follow_up = "Do any of these catch your eye?"
        
        return f"{preamble}{recommendation_text} {follow_up}"
    
    return "Here are some options for you."


# --- Main Message Processing Function ---
async def process_user_message(
    conversation_history: List[Dict[str, str]], 
    user_message: str,
    user_context: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Processes user message with user profile context.
    """
    logger.info(f"Processing user message: '{user_message}'")
    if user_context:
        logger.debug(f"User context: {user_context}")

    # 1. Determine Next Action with user context
    action_prompt_messages = _construct_conversation_prompt(
        conversation_history, 
        user_message, 
        user_context
    )
    
    llm_action_response_text = await generate_text(
        messages=action_prompt_messages,
        model=CONVERSATIONAL_MODEL_NAME,
        temperature=0.1,
        max_tokens=800
    )
    
    action_data = _parse_llm_action_response(llm_action_response_text)

    if not action_data:
        logger.error("Failed to determine next action from LLM parsing.")
        return {
            "text_response": FALLBACK_ERROR_MESSAGE, 
            "products": [],
            "extracted_criteria": {}
        }

    action = action_data.get("action")
    extracted_criteria = action_data.get("extracted_criteria", {})
    
    # Enhance criteria with user profile defaults
    if action == "TRIGGER_RETRIEVAL" and user_profile:
        extracted_criteria = _enhance_criteria_with_profile(extracted_criteria, user_profile)
        action_data["extracted_criteria"] = extracted_criteria
    
    final_text_response = ""
    recommended_products = []

    # 2. Execute Action
    if action == "ASK_CLARIFICATION":
        # Personalize clarification questions based on profile
        clarification = action_data.get("clarification_question", FALLBACK_ERROR_MESSAGE)
        
        # Add personalized touch if applicable
        if user_profile and user_profile.get("interaction_count", 0) > 5:
            prefs = user_profile.get("preferences", {})
            if prefs.get("colors") and "color" in clarification.lower():
                top_color = max(prefs["colors"].items(), key=lambda x: x[1])[0]
                clarification += f" (I know you often like {top_color})"
        
        final_text_response = clarification
        logger.info(f"Action: Ask Clarification - '{final_text_response}'")

    elif action == "ANSWER_QUESTION" or action == "CHIT_CHAT":
        final_text_response = action_data.get("answer_text", FALLBACK_ERROR_MESSAGE)
        logger.info(f"Action: Answer/ChitChat - '{final_text_response[:100]}...'")

    elif action == "COMPARE_PRODUCTS":
        logger.info(f"Action: Compare Products - Items: {action_data.get('products_to_compare')}")
        product_names_to_compare = action_data.get("products_to_compare", [])

        if len(product_names_to_compare) < 2:
            final_text_response = "Please specify at least two products you'd like me to compare."
            recommended_products = []
        else:
            # Retrieve data for the specified products
            product_map = {
                p['name'].strip().lower(): p
                for p in _product_data_store if isinstance(p.get('name'), str)
            }
            found_products_data = []
            missing_names = []
            for name in product_names_to_compare:
                product_data = product_map.get(name.lower().strip())
                if product_data:
                    found_products_data.append(product_data)
                else:
                    missing_names.append(name)

            if missing_names:
                logger.warning(f"Could not find product data for comparison: {missing_names}")
                if len(found_products_data) >= 2:
                    final_text_response = f"Sorry, I couldn't find details for {', '.join(missing_names)}. I can compare the ones I found: {', '.join([p['name'] for p in found_products_data])}."
                    products_to_compare_data = found_products_data
                elif len(found_products_data) == 1:
                    final_text_response = f"Sorry, I couldn't find details for {', '.join(missing_names)}. I only found {found_products_data[0]['name']}. Please mention at least two products I know for comparison."
                    products_to_compare_data = []
                else:
                    final_text_response = f"Sorry, I couldn't find details for any of these products: {', '.join(missing_names)}. Could you please check the names?"
                    products_to_compare_data = []
            else:
                products_to_compare_data = found_products_data

            # If we have at least two products, generate the comparison
            if len(products_to_compare_data) >= 2:
                comparison_prompt_messages = _build_comparison_prompt(products_to_compare_data, user_message)
                logger.debug("Calling LLM for factual comparison generation...")
                comparison_response_text = await generate_text(
                    messages=comparison_prompt_messages,
                    model=CONVERSATIONAL_MODEL_NAME,
                    temperature=0.1,
                    max_tokens=1000
                )

                if comparison_response_text:
                    if final_text_response:
                        final_text_response += "\n\nHere's the comparison for the ones I found:\n" + comparison_response_text
                    else:
                        final_text_response = comparison_response_text
                    recommended_products = products_to_compare_data
                else:
                    logger.error("Comparison LLM call failed.")
                    final_text_response = f"Sorry, I couldn't generate the comparison for {', '.join([p['name'] for p in products_to_compare_data])} right now."
                    recommended_products = []
            else:
                recommended_products = []

    elif action == "TRIGGER_RETRIEVAL":
        retrieval_query = action_data.get("retrieval_query")
        if not retrieval_query:
            logger.error("Action was TRIGGER_RETRIEVAL, but 'retrieval_query' was missing.")
            final_text_response = FALLBACK_ERROR_MESSAGE
        elif not is_retrieval_initialized():
            logger.error("Action TRIGGER_RETRIEVAL, but retrieval system not initialized.")
            final_text_response = "I'm sorry, my product search system isn't quite ready. Please try again shortly."
        else:
            logger.info(f"Action: Trigger Retrieval - Query: '{retrieval_query}', Criteria: {extracted_criteria}")

            candidate_products, relaxation_info = await get_product_candidates_hybrid(retrieval_query, extracted_criteria)

            if candidate_products:
                log_suffix = f"after relaxing {relaxation_info}" if relaxation_info else "with strict criteria"
                logger.info(f"Retrieved {len(candidate_products)} candidates {log_suffix}. Proceeding to reranking.")

                current_turn_history = conversation_history + [{"role": "user", "content": user_message}]
                recommended_products = await reflect_and_rerank(current_turn_history, candidate_products)

                if recommended_products:
                    # Check for mismatches
                    mismatched_criteria_messages = []
                    
                    # Check Color Mismatch
                    requested_color = extracted_criteria.get("color")
                    if requested_color:
                        color_found = False
                        for product in recommended_products:
                            product_color_val = product.get("color")
                            if product_color_val:
                                product_colors_lower = [c.strip() for c in product_color_val.lower().split('/')]
                                if requested_color.lower().strip() in product_colors_lower:
                                    color_found = True
                                    break
                        if not color_found:
                            mismatched_criteria_messages.append(f"in the exact color '{requested_color}'")
                            logger.info(f"Mismatch detected: Requested color '{requested_color}' not found in recommendations.")

                    # Check Material Mismatch
                    requested_material = extracted_criteria.get("material_preference")
                    if requested_material and requested_material not in ['any', None]:
                        material_found = False
                        real_leathers = ['full-grain', 'top-grain']
                        vegan_leathers = ['vegan leather']
                        synthetics = ['leatherette', 'polyester']
                        for product in recommended_products:
                            product_leather = product.get('leather_type', '').lower()
                            is_real = any(rl in product_leather for rl in real_leathers)
                            is_vegan = any(vl in product_leather for vl in vegan_leathers)
                            is_synthetic = any(syn in product_leather for syn in synthetics)
                            if requested_material == 'real_leather' and is_real: 
                                material_found = True
                                break
                            if requested_material == 'vegan' and is_vegan: 
                                material_found = True
                                break
                            if requested_material == 'synthetic' and is_synthetic: 
                                material_found = True
                                break
                        if not material_found:
                            mismatched_criteria_messages.append(f"with the material '{requested_material}'")
                            logger.info(f"Mismatch detected: Requested material '{requested_material}' not found in recommendations.")

                    # Generate personalized response
                    final_text_response = _generate_personalized_response(
                        "recommendations",
                        recommended_products,
                        user_profile,
                        relaxation_info,
                        mismatched_criteria_messages if mismatched_criteria_messages else None
                    )
                    logger.info(f"Successfully reranked. Recommending {len(recommended_products)} products.")
                else:
                    logger.warning("Reranking failed or returned no products after successful retrieval/relaxation.")
                    key_features = ["color", "material", "style", "a specific feature"]
                    relax_context = 'after relaxing criteria' if relaxation_info else 'based on your request'
                    final_text_response = f"I found some potential matches {relax_context}, but need a bit more guidance to narrow it down. Is there a particular feature that's most important to you right now, like the {random.choice(key_features)}?"
                    recommended_products = []

            else:
                logger.info("Retrieval returned no candidates, even after attempting relaxation.")
                response_intro = f"Hmm, I searched based on '{retrieval_query}' but couldn't find a match, even after trying to relax the criteria like sustainability or price a bit."
                suggestions = []
                if extracted_criteria.get("min_sustainability", 0) >= 7.0: 
                    suggestions.append("significantly lower the required sustainability level?")
                if extracted_criteria.get("material_preference") == "vegan": 
                    suggestions.append("consider alternative materials like high-quality leatherette or even real leather?")
                elif extracted_criteria.get("material_preference") == "real_leather": 
                    suggestions.append("consider synthetic options like vegan leather or leatherette?")
                if extracted_criteria.get("style"): 
                    suggestions.append(f"look at different styles beyond '{extracted_criteria.get('style')}'?")
                if extracted_criteria.get("price_min") or extracted_criteria.get("price_max"): 
                    suggestions.append("adjust the price range?")
                if not suggestions: 
                    suggestions.append("try describing your needs differently?")
                final_text_response = f"{response_intro} Would you like to {random.choice(suggestions)}"
                recommended_products = []

    else:
        logger.error(f"LLM returned unknown or unhandled action: {action}")
        final_text_response = FALLBACK_ERROR_MESSAGE
        recommended_products = []

    # 3. Ensure products are serializable
    serializable_products = []
    if recommended_products:
        for p in recommended_products:
            if isinstance(p, dict):
                try:
                    json.dumps(p)
                    serializable_products.append(p)
                except TypeError:
                    logger.warning(f"Product ID {p.get('id', 'N/A')} contains non-serializable data, attempting basic conversion.")
                    converted_p = {}
                    for k, v in p.items():
                        if hasattr(v, 'item'):  # Basic check for numpy types
                            try: 
                                converted_p[k] = v.item()
                            except: 
                                converted_p[k] = str(v)
                        else: 
                            converted_p[k] = v
                    try:
                        json.dumps(converted_p)
                        serializable_products.append(converted_p)
                    except TypeError:
                        logger.error(f"Could not serialize product ID {p.get('id', 'N/A')} even after basic conversion. Skipping.")

        if len(serializable_products) != len(recommended_products):
            logger.warning("Some recommended products were not serializable and were filtered out.")

    # 4. Return structured response with extracted criteria
    return {
        "text_response": final_text_response if final_text_response else FALLBACK_ERROR_MESSAGE,
        "products": serializable_products,
        "extracted_criteria": extracted_criteria  # Include for profile updates
    }