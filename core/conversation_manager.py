# luxury_leather_chatbot/core/conversation_manager.py

import logging
import json
import random
import copy
from typing import List, Dict, Any, Optional
import json # Ensure json is imported

# --- Absolute Imports ---
from core.llm_interface import generate_text
# Need access to product data store for comparison details
from core.retrieval_module import (
    get_product_candidates_hybrid,
    is_retrieval_initialized,
    _product_data_store # Import the global data store
)
from core.reranking_module import reflect_and_rerank
from config import (
    CONVERSATIONAL_MODEL_NAME, FALLBACK_ERROR_MESSAGE, RERANKING_TOP_N,
    RERANKING_MODEL_NAME # Added in case needed for comparison
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Helper Functions (Prompts & Parsing) ---

def _construct_conversation_prompt(conversation_history: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    """
    Constructs the prompt for the conversational LLM to determine action,
    extract criteria, OR identify products for comparison.
    """
    # --- FULL UPDATED SYSTEM PROMPT ---
    system_prompt = f"""You are a helpful and knowledgeable personal shopper for a luxury leather brand.
Your goal is to understand the user's needs through conversation and assist them effectively.

Analyze the conversation history and the latest user message. Determine the user's primary intent.

**Possible Intents & Actions:**

1.  **Find Products (Action: `TRIGGER_RETRIEVAL`)**:
    *   If the user is asking for recommendations or searching for products with specific features.
    *   **Requires *reasonably specific* criteria.** Ideally, this includes style AND color, occasion, price range, or specific features. Do NOT trigger retrieval if only a broad style (like 'briefcase', 'bag') is mentioned without further context or details from the user message or recent history.
    *   Extract key criteria (style, color, material, occasion, price, quality scores) using the specified format below.
    *   Formulate a concise `retrieval_query` summarizing the core need for semantic search.

2.  **Compare Specific Products (Action: `COMPARE_PRODUCTS`)**:
    *   If the user explicitly asks to compare specific products mentioned recently in the conversation (e.g., "compare X and Y", "what's the difference between A, B, C?", "tell me more about X vs Y").
    *   Extract the **exact names** of the products mentioned for comparison into the `products_to_compare` list. ONLY include names that seem to be actual product names from the brand.

3.  **Ask for Clarification (Action: `ASK_CLARIFICATION`)**:
    *   **Trigger this if the user provides only general criteria (like just the style or a vague use case) and more information would significantly help narrow down choices.** For example, if they just say "I want a briefcase", ask about color, material, or intended use.
    *   Also trigger if crucial information for search is missing (e.g., no style or context provided at all) or criteria are ambiguous/conflicting.
    *   Ask ONE targeted question to get the most helpful next piece of information (e.g., "Okay, a briefcase! What color are you thinking of?" or "Got it. Are you looking for a briefcase primarily for work, travel, or something else?").

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
- Ensure the output is always valid JSON. Do not add explanations outside the JSON object.
"""
    # --- End of System Prompt ---

    messages = [{"role": "system", "content": system_prompt}]
    # Limit history passed to this initial prompt if it gets too long
    MAX_HISTORY_FOR_ACTION = 6 # Keep last 3 user/assistant turns for context
    messages.extend(conversation_history[-(MAX_HISTORY_FOR_ACTION*2):])
    messages.append({"role": "user", "content": user_message})

    return messages


def _parse_llm_action_response(response_text: str) -> Optional[Dict[str, Any]]:
    # --- MODIFIED to validate COMPARE_PRODUCTS action ---
    if not response_text: logger.error("Received empty response from conversational LLM."); return None
    try:
        # Find JSON block robustly
        json_start = response_text.find('{'); json_end = response_text.rfind('}')
        if json_start == -1 or json_end == -1 or json_end < json_start:
            logger.error(f"Could not find valid JSON object delimiters in LLM action response: {response_text}")
            try: action_data = json.loads(response_text.strip()) # Try direct load
            except json.JSONDecodeError: logger.error(f"Direct JSON loading failed for response: {response_text}"); return None
        else: json_str = response_text[json_start:json_end+1]; action_data = json.loads(json_str)

        logger.debug(f"Parsed LLM action data: {action_data}")

        # Basic validation
        if "action" not in action_data: logger.error(f"LLM action missing 'action' key: {action_data}"); return None
        action = action_data["action"]

        # Action-specific validation
        if action == "TRIGGER_RETRIEVAL":
            if "extracted_criteria" not in action_data: action_data["extracted_criteria"] = {}; logger.warning("LLM action TRIGGER_RETRIEVAL missing 'extracted_criteria', adding empty dict.")
            if not isinstance(action_data["extracted_criteria"], dict): logger.error(f"'extracted_criteria' not a dict: {action_data['extracted_criteria']}"); action_data["extracted_criteria"] = {}
            if not action_data.get("retrieval_query"): logger.error("Action TRIGGER_RETRIEVAL missing 'retrieval_query'."); return None
            # Optional: Validate style within criteria
            if action_data.get("extracted_criteria", {}).get("style"):
                allowed_styles = {"briefcase", "messenger", "tote", "duffel", "handbag", "slim bag", "sling", "crossbody", "travel bag", "trolley"}
                extracted_style = action_data["extracted_criteria"]["style"].lower()
                if extracted_style not in allowed_styles: logger.warning(f"LLM extracted an invalid style: '{extracted_style}'.")

        elif action == "COMPARE_PRODUCTS":
            if "products_to_compare" not in action_data: logger.error("Action COMPARE_PRODUCTS missing 'products_to_compare'."); return None
            if not isinstance(action_data.get("products_to_compare"), list): logger.error("Action COMPARE_PRODUCTS: 'products_to_compare' is not a list."); return None
            if not all(isinstance(name, str) for name in action_data["products_to_compare"]): logger.error("Action COMPARE_PRODUCTS: 'products_to_compare' contains non-string elements."); return None
            if len(action_data["products_to_compare"]) < 2: logger.warning("Action COMPARE_PRODUCTS: Fewer than 2 products listed for comparison."); # Allow for now, handle later

        elif action == "ASK_CLARIFICATION":
            if not action_data.get("clarification_question"): logger.error("Action ASK_CLARIFICATION missing 'clarification_question'."); return None

        elif action == "ANSWER_QUESTION" or action == "CHIT_CHAT":
            if not action_data.get("answer_text"): logger.error(f"Action {action} missing 'answer_text'."); return None

        # Ensure extracted_criteria exists even if not TRIGGER_RETRIEVAL for safety
        if "extracted_criteria" not in action_data: action_data["extracted_criteria"] = {}

        return action_data
    except json.JSONDecodeError as e: logger.error(f"Failed decode LLM action JSON: {e}", exc_info=True); logger.debug(f"Problematic text: {response_text}"); return None
    except Exception as e: logger.error(f"Unexpected error parsing LLM action response: {e}", exc_info=True); return None


# --- Function to build the comparison prompt ---
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
        # *** FIX: Use p_data here ***
        product_details_text += f"Sustainability Score: {p_data.get('sustainability_score', 'N/A')}\n"
        # *** FIX: Use p_data here ***
        product_details_text += f"Occasion(s): {p_data.get('occasion', 'Not specified')}\n"
        product_details_text += f"Description: {p_data.get('description', 'No description provided.')}\n"
        product_details_text += "---\n"

    full_prompt = prompt_intro + product_details_text + "\nComparison Task: Generate the comparison now."

    return [{"role": "user", "content": full_prompt}]


# --- Main Message Processing Function ---
async def process_user_message(conversation_history: List[Dict[str, str]], user_message: str) -> Dict[str, Any]:
    """
    Processes user message, orchestrates LLM action determination, retrieval/comparison,
    and response generation.
    """
    logger.info(f"Processing user message: '{user_message}'")
    logger.debug(f"Incoming history length: {len(conversation_history)}")

    # 1. Determine Next Action
    action_prompt_messages = _construct_conversation_prompt(conversation_history, user_message)
    llm_action_response_text = await generate_text(
        messages=action_prompt_messages,
        model=CONVERSATIONAL_MODEL_NAME, # Use the conversational model for action determination
        temperature=0.1, # Lower temp for more deterministic action/extraction
        max_tokens=800 # Needs enough space for criteria/names list
    )
    action_data = _parse_llm_action_response(llm_action_response_text)

    if not action_data:
        logger.error("Failed to determine next action from LLM parsing.")
        return {"text_response": FALLBACK_ERROR_MESSAGE, "products": []}

    action = action_data.get("action")
    extracted_criteria = action_data.get("extracted_criteria", {})
    final_text_response = ""
    recommended_products = [] # Usually empty unless retrieving/comparing

    # 2. Execute Action
    if action == "ASK_CLARIFICATION":
        final_text_response = action_data.get("clarification_question", FALLBACK_ERROR_MESSAGE)
        logger.info(f"Action: Ask Clarification - '{final_text_response}'")

    elif action == "ANSWER_QUESTION" or action == "CHIT_CHAT":
        # Use the LLM's generated answer, assuming it's general knowledge
        final_text_response = action_data.get("answer_text", FALLBACK_ERROR_MESSAGE)
        logger.info(f"Action: Answer/ChitChat - '{final_text_response[:100]}...'")

    elif action == "COMPARE_PRODUCTS":
        logger.info(f"Action: Compare Products - Items: {action_data.get('products_to_compare')}")
        product_names_to_compare = action_data.get("products_to_compare", [])

        if len(product_names_to_compare) < 2:
            final_text_response = "Please specify at least two products you'd like me to compare."
            recommended_products = [] # Ensure empty
        else:
            # Retrieve data for the specified products
            # Create a quick lookup map from the global store
            # TODO: Implement fuzzy matching here for robustness
            product_map = {
            p['name'].strip().lower(): p
            for p in _product_data_store if isinstance(p.get('name'), str)
            }
            found_products_data = []
            missing_names = []
            for name in product_names_to_compare:
                product_data = product_map.get(name.lower().strip()) # Match case-insensitively
                if product_data:
                    found_products_data.append(product_data)
                else:
                    missing_names.append(name)

            if missing_names:
                logger.warning(f"Could not find product data for comparison: {missing_names}")
                # Inform user which products were not found
                if len(found_products_data) >= 2:
                     final_text_response = f"Sorry, I couldn't find details for {', '.join(missing_names)}. I can compare the ones I found: {', '.join([p['name'] for p in found_products_data])}."
                     products_to_compare_data = found_products_data
                elif len(found_products_data) == 1:
                     final_text_response = f"Sorry, I couldn't find details for {', '.join(missing_names)}. I only found {found_products_data[0]['name']}. Please mention at least two products I know for comparison."
                     products_to_compare_data = [] # Prevent comparison call
                else:
                     final_text_response = f"Sorry, I couldn't find details for any of these products: {', '.join(missing_names)}. Could you please check the names?"
                     products_to_compare_data = [] # Prevent comparison call

            else:
                # All products found, proceed with comparison
                products_to_compare_data = found_products_data

            # If we have at least two products, generate the comparison
            if len(products_to_compare_data) >= 2:
                comparison_prompt_messages = _build_comparison_prompt(products_to_compare_data, user_message)
                logger.debug("Calling LLM for factual comparison generation...")
                comparison_response_text = await generate_text(
                    messages=comparison_prompt_messages,
                    # Use a model good at following instructions
                    model=CONVERSATIONAL_MODEL_NAME, # Or RERANKING_MODEL_NAME
                    temperature=0.1, # Low temp for factual consistency
                    max_tokens=1000 # Allow enough space for comparison
                )

                if comparison_response_text:
                    # Combine preamble (if needed) + comparison text
                    if final_text_response: # Prepend the "Sorry, I couldn't find..." message if applicable
                         final_text_response += "\n\nHere's the comparison for the ones I found:\n" + comparison_response_text
                    else:
                         final_text_response = comparison_response_text
                    # *** Populate products for comparison action ***
                    recommended_products = products_to_compare_data # Pass the compared product data back
                else:
                    logger.error("Comparison LLM call failed.")
                    final_text_response = f"Sorry, I couldn't generate the comparison for {', '.join([p['name'] for p in products_to_compare_data])} right now."
                    recommended_products = [] # Ensure empty if comparison failed

            # If only 0 or 1 product was found after checking names, final_text_response is already set above
            else:
                 recommended_products = [] # Ensure empty if not enough products to compare

    elif action == "TRIGGER_RETRIEVAL":
        retrieval_query = action_data.get("retrieval_query")
        if not retrieval_query:
            logger.error("Action was TRIGGER_RETRIEVAL, but 'retrieval_query' was missing.")
            final_text_response = FALLBACK_ERROR_MESSAGE
            recommended_products = []
        elif not is_retrieval_initialized():
             logger.error("Action TRIGGER_RETRIEVAL, but retrieval system not initialized.")
             final_text_response = "I'm sorry, my product search system isn't quite ready. Please try again shortly."
             recommended_products = []
        else:
            logger.info(f"Action: Trigger Retrieval - Query: '{retrieval_query}', Criteria: {extracted_criteria}")

            candidate_products, relaxation_info = await get_product_candidates_hybrid(retrieval_query, extracted_criteria)

            if candidate_products:
                log_suffix = f"after relaxing {relaxation_info}" if relaxation_info else "with strict criteria"
                logger.info(f"Retrieved {len(candidate_products)} candidates {log_suffix}. Proceeding to reranking.") # candidate_products here is the list returned by get_product_candidates_hybrid

                current_turn_history = conversation_history + [{"role": "user", "content": user_message}]
                recommended_products = await reflect_and_rerank(current_turn_history, candidate_products) # Pass the retrieved candidates to rerank

                if recommended_products:
                    # --- START: Conversational Mismatch Check ---
                    num_products = len(recommended_products)
                    product_names = [p.get('name', 'Unknown') for p in recommended_products]

                    mismatch_preamble = ""
                    mismatched_criteria_messages = []

                    # 1. Check Color Mismatch
                    requested_color = extracted_criteria.get("color")
                    if requested_color:
                        color_found = False
                        for product in recommended_products:
                            product_color_val = product.get("color")
                            if product_color_val:
                                product_colors_lower = [c.strip() for c in product_color_val.lower().split('/')]
                                if requested_color.lower().strip() in product_colors_lower:
                                    color_found = True
                                    break # Found at least one product with the requested color
                        if not color_found:
                            mismatched_criteria_messages.append(f"in the exact color '{requested_color}'")
                            logger.info(f"Mismatch detected: Requested color '{requested_color}' not found in recommendations.")

                    # 2. Check Style Mismatch (Could be added if needed)
                    # requested_style = extracted_criteria.get("style")
                    # if requested_style:
                    #    # Check if at least one recommended product matches the style
                    #    # ... logic ...
                    #    if not style_found: mismatched_criteria_messages.append(f"in the specific style '{requested_style}'")

                    # 3. Check Material Mismatch
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
                             if requested_material == 'real_leather' and is_real: material_found = True; break
                             if requested_material == 'vegan' and is_vegan: material_found = True; break
                             if requested_material == 'synthetic' and is_synthetic: material_found = True; break
                         if not material_found:
                             mismatched_criteria_messages.append(f"with the material '{requested_material}'")
                             logger.info(f"Mismatch detected: Requested material '{requested_material}' not found in recommendations.")

                    # Construct the preamble if mismatches were found
                    if mismatched_criteria_messages:
                        mismatch_details = " and ".join(mismatched_criteria_messages)
                        preamble_options = [
                            f"Okay, I couldn't find anything available right now {mismatch_details}, but based on your other preferences, these might be good alternatives: ",
                            f"While I don't have an exact match {mismatch_details}, these options are quite relevant otherwise and might interest you: ",
                            f"Hmm, nothing matched perfectly {mismatch_details}. However, take a look at these suggestions: "
                        ]
                        mismatch_preamble = random.choice(preamble_options)
                    # --- END: Conversational Mismatch Check ---


                    # --- Build the rest of the response ---
                    effective_preamble = mismatch_preamble # Use mismatch preamble if generated
                    if not effective_preamble and relaxation_info: # Otherwise, use relaxation preamble if applicable
                         relaxation_preamble_options = [ f"Okay, I couldn't find an exact match for *all* your preferences, but by slightly relaxing the **{relaxation_info}**, ", f"While nothing matched perfectly, adjusting the **{relaxation_info}** allowed me to find these options: ", f"To find some relevant choices, I relaxed the **{relaxation_info}** a bit. " ]
                         effective_preamble = random.choice(relaxation_preamble_options)

                    ack_phrases = ["Alright!", "Okay!", "Got it.", "Let's see..."] if not effective_preamble else ["", "Here's what I found: ", "Take a look: "]

                    # Build criteria hints, EXCLUDING mismatched criteria
                    criteria_hints = []
                    style_crit=extracted_criteria.get("style"); material_crit=extracted_criteria.get("material_preference"); price_min_crit=extracted_criteria.get("price_min"); price_max_crit=extracted_criteria.get("price_max"); sustain_crit=extracted_criteria.get("min_sustainability", 0)
                    color_crit = extracted_criteria.get("color")

                    if style_crit: criteria_hints.append(f"for those {style_crit}s")

                    if material_crit == "vegan" and "material 'vegan'" not in " ".join(mismatched_criteria_messages): criteria_hints.append("for vegan options")
                    elif material_crit == "real_leather" and "material 'real_leather'" not in " ".join(mismatched_criteria_messages): criteria_hints.append("for real leather")

                    if color_crit and f"color '{color_crit}'" not in " ".join(mismatched_criteria_messages):
                         criteria_hints.append(f"in {color_crit}")

                    if price_min_crit and price_max_crit: criteria_hints.append(f"around ${price_min_crit}-${price_max_crit}")
                    if sustain_crit >= 6.5: criteria_hints.append("with good sustainability focus")

                    ack_detail = ""; chosen_hints = list(set(criteria_hints));
                    if len(chosen_hints) >= 2: ack_detail = f" ({random.choice(chosen_hints[:2])})"
                    elif chosen_hints: ack_detail = f" ({chosen_hints[0]})"

                    # Build recommendation list string
                    if num_products == 1:
                        product_list_str = f"the '{product_names[0]}'"
                        recommendation_intro_phrase = random.choice([ f"I found this option{ack_detail} you might like: {product_list_str}.", f"how about this one{ack_detail}? {product_list_str}.", f"this one{ack_detail} stood out: {product_list_str}."])
                        follow_up_question = random.choice([ "Does this seem like a good alternative?", "What do you think of this one?", "Would you like more details about it, even though the criteria aren't exact?" ]) if mismatched_criteria_messages else random.choice([ "Does this seem like a good fit?", "What do you think?", "Would you like more details about it?" ])
                    else:
                        product_list_str = ", ".join([f"'{name}'" for name in product_names[:-1]]) + f", and '{product_names[-1]}'"
                        recommendation_intro_phrase = random.choice([ f"I found these {num_products} options{ack_detail} you might like: {product_list_str}.", f"how about these{ack_detail}? {product_list_str}." ])
                        follow_up_question = random.choice([ "Do any of these seem like good alternatives?", "What do you think of these?", "Would you like more details about any of them, even though they aren't an exact match?" ]) if mismatched_criteria_messages else random.choice([ "Do any of these catch your eye?", "What do you think of these?", "Would you like more details about any of them?" ])

                    final_text_response = f"{effective_preamble}{random.choice(ack_phrases)}{recommendation_intro_phrase} {follow_up_question}"
                    logger.info(f"Successfully reranked. Recommending {len(recommended_products)} products. Mismatches acknowledged: {bool(mismatched_criteria_messages)}")

                else: # Reranking failed or returned no products
                    logger.warning("Reranking failed or returned no products after successful retrieval/relaxation.")
                    key_features = ["color", "material", "style", "a specific feature"]
                    relax_context = 'after relaxing criteria' if relaxation_info else 'based on your request'
                    final_text_response = f"I found some potential matches {relax_context}, but need a bit more guidance to narrow it down. Is there a particular feature that's most important to you right now, like the {random.choice(key_features)}?"
                    recommended_products = [] # Keep empty

            else: # Retrieval found nothing, even after relaxation
                 logger.info("Retrieval returned no candidates, even after attempting relaxation.")
                 response_intro = f"Hmm, I searched based on '{retrieval_query}' but couldn't find a match, even after trying to relax the criteria like sustainability or price a bit."
                 suggestions = []
                 # ... (suggestion logic remains same) ...
                 if extracted_criteria.get("min_sustainability", 0) >= 7.0: suggestions.append("significantly lower the required sustainability level?")
                 if extracted_criteria.get("material_preference") == "vegan": suggestions.append("consider alternative materials like high-quality leatherette or even real leather?")
                 elif extracted_criteria.get("material_preference") == "real_leather": suggestions.append("consider synthetic options like vegan leather or leatherette?")
                 if extracted_criteria.get("style"): suggestions.append(f"look at different styles beyond '{extracted_criteria.get('style')}'?")
                 if extracted_criteria.get("price_min") or extracted_criteria.get("price_max"): suggestions.append("adjust the price range?")
                 if not suggestions: suggestions.append("try describing your needs differently?")
                 final_text_response = f"{response_intro} Would you like to {random.choice(suggestions)}"
                 recommended_products = [] # Ensure products list is empty

    else: # Handle unknown action
        logger.error(f"LLM returned unknown or unhandled action: {action}")
        final_text_response = FALLBACK_ERROR_MESSAGE
        recommended_products = [] # Ensure empty

    # 3. Update History & Return structured response
    # NOTE: History update logic is now handled in chat_routes.py using session

    # Ensure products are serializable (should contain dicts)
    serializable_products = []
    if recommended_products:
         for p in recommended_products:
             if isinstance(p, dict):
                  # Ensure all values within the dict are JSON serializable (e.g., no numpy types)
                  # A simple check, might need more robust serialization if complex objects are nested
                  try:
                       json.dumps(p) # Test serialization
                       serializable_products.append(p)
                  except TypeError:
                       logger.warning(f"Product ID {p.get('id', 'N/A')} contains non-serializable data, attempting basic conversion.")
                       # Attempt basic conversion (e.g., numpy floats to python floats)
                       # This part might need refinement based on actual data types encountered
                       converted_p = {}
                       for k, v in p.items():
                           if hasattr(v, 'item'): # Basic check for numpy types
                               try: converted_p[k] = v.item()
                               except: converted_p[k] = str(v) # Fallback to string
                           else: converted_p[k] = v
                       try:
                            json.dumps(converted_p)
                            serializable_products.append(converted_p)
                       except TypeError:
                            logger.error(f"Could not serialize product ID {p.get('id', 'N/A')} even after basic conversion. Skipping.")

         if len(serializable_products) != len(recommended_products):
             logger.warning("Some recommended products were not serializable and were filtered out.")

    # Prepare the final response payload
    return {
        "text_response": final_text_response if final_text_response else FALLBACK_ERROR_MESSAGE, # Ensure fallback if text is empty
        "products": serializable_products # Will be populated for retrieval/comparison
    }