#core/reranking_module
import json
import logging
from typing import List, Dict, Any, Optional

# --- Absolute Imports ---
from core.llm_interface import generate_text
from config import RERANKING_MODEL_NAME, RERANKING_TOP_N

# --- Logging ---
logger = logging.getLogger(__name__)


def _format_products_for_prompt(products: List[Dict[str, Any]]) -> str:
    """Formats product details clearly for the LLM reranking prompt."""
    output = ""
    if not products:
        return "[No products to display]"

    for i, p in enumerate(products):
        output += f"\n--- Product {i+1} ---\n"
        output += f"ID: {p.get('id')}\n"
        output += f"Name: {p.get('name')}\n"
        output += f"Style: {p.get('style', 'N/A')}\n"
        output += f"Color: {p.get('color', 'N/A')}\n"
        output += f"Leather Type: {p.get('leather_type', 'N/A')}\n"
        output += f"Occasion: {p.get('occasion', 'N/A')}\n"
        output += f"Price: ${p.get('price', 0.0):.2f}\n" # Format price
        output += f"Craftsmanship Score: {p.get('craftsmanship_score', 'N/A')}\n"
        # output += f"Sustainability Score: {p.get('sustainability_score', 'N/A')}\n" # Optional
        output += f"Description: {p.get('description', '')[:200]}...\n" # Limit description length
    output += "---\n"
    return output

def _construct_rerank_prompt(conversation_history: List[Dict[str, str]], candidate_products: List[Dict[str, Any]]) -> str:
    """Constructs the prompt for the LLM to rate products."""

    convo_string = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history])
    products_string = _format_products_for_prompt(candidate_products)
    candidate_ids = [p.get('id') for p in candidate_products]

    prompt = f"""
You are an expert luxury leather goods recommender acting as a critical evaluator.
Your task is to evaluate a list of candidate products based on a user's conversation history and assign a relevance score to EACH product.

**Conversation History:**
{convo_string}

**Candidate Products to Evaluate (IDs: {candidate_ids}):**
{products_string}

**Evaluation Criteria:**
Carefully assess each product against the user's stated and implied preferences revealed in the conversation (style, color, occasion, price sensitivity, specific features mentioned).
Consider additional quality factors:
- **Craftsmanship Score:** Higher scores are generally better.
- **Suitability:** How well does the product fit the overall context and specific needs?
- **Relevance:** Prioritize direct relevance to the user's request above all else.

**Rating Scale:**
Assign an integer score from -2 to +2 to EACH product ID based on the criteria:
+2: Excellent Match - Strongly aligns with user needs and desirable attributes. High relevance and quality.
+1: Good Match - Meets most key criteria well. Relevant and decent quality.
 0: Neutral/Okay Match - Meets some criteria but has notable drawbacks, isn't a standout, or relevance is ambiguous. Might be technically relevant but not ideal.
-1: Poor Match - Significant mismatch with one or more key criteria. Low relevance.
-2: Very Poor Match / Irrelevant - Clearly unsuitable or irrelevant based on the conversation.

**Output Format:**
Return ONLY a single JSON object where keys are the product IDs (as strings) and values are the integer ratings you assigned. Example format:
{{
  "product_id_1": 1,
  "product_id_2": -1,
  "product_id_3": 2,
  ...
  "product_id_N": 0
}}

**Instructions:**
- Ensure EVERY product ID from the candidate list ({candidate_ids}) is included as a key in the JSON object exactly once.
- The value for each key MUST be an integer between -2 and 2.
- Do NOT include any text, explanations, or markdown formatting before or after the JSON object. Just the raw JSON.
"""
    return prompt

def _parse_and_validate_ratings(llm_response: str, candidate_ids: List[int]) -> Optional[Dict[int, int]]:
    """Parses the LLM response, validates it, and returns ratings."""
    if not llm_response:
        logger.error("LLM reranking response was empty.")
        return None
    try:
        # Clean potential markdown fences or other unwanted text
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}')
        if json_start == -1 or json_end == -1:
            logger.error(f"Could not find JSON object in LLM response: {llm_response}")
            return None
        json_str = llm_response[json_start:json_end+1]

        ratings_raw = json.loads(json_str)

        if not isinstance(ratings_raw, dict):
            logger.error(f"LLM Reranking output is not a dictionary: {json_str}")
            return None

        validated_ratings = {}
        candidate_id_set = set(candidate_ids)
        returned_id_set = set()

        for pid_str, rating in ratings_raw.items():
            try:
                pid = int(pid_str)
                returned_id_set.add(pid)

                if pid not in candidate_id_set:
                    logger.warning(f"LLM returned rating for unexpected product ID: {pid}. Ignoring.")
                    continue

                if not isinstance(rating, int) or not (-2 <= rating <= 2):
                    logger.warning(f"Invalid rating value '{rating}' for product ID {pid}. Assigning 0.")
                    validated_ratings[pid] = 0
                else:
                    validated_ratings[pid] = rating

            except ValueError:
                logger.warning(f"Could not convert product ID key '{pid_str}' to int. Skipping.")
                continue
            except Exception as e:
                 logger.error(f"Error processing rating for '{pid_str}': {e}", exc_info=True)
                 continue

        # Check if all candidate IDs were rated, assign 0 to missing ones
        if returned_id_set != candidate_id_set:
             missing_ids = candidate_id_set - returned_id_set
             logger.warning(f"LLM did not return ratings for all candidate IDs. Missing: {missing_ids}. Assigning 0.")
             for missing_id in missing_ids:
                 validated_ratings[missing_id] = 0

        if not validated_ratings:
             logger.error("No valid ratings could be extracted from the LLM response.")
             return None

        return validated_ratings

    except json.JSONDecodeError:
        logger.error(f"Failed to decode LLM reranking response into JSON: {json_str}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing ratings: {e}", exc_info=True)
        return None


async def reflect_and_rerank(
    conversation_history: List[Dict[str, str]],
    candidate_products: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reranks candidate products using LLM reflection and rating.

    Args:
        conversation_history: The current conversation context.
        candidate_products: List of product dictionaries retrieved by db-ally.

    Returns:
        The top N (RERANKING_TOP_N) products, ordered by rating,
        or the original top N if reranking fails.
    """
    if not candidate_products:
        logger.info("No candidate products provided for reranking.")
        return []

    # Fallback: original top N products
    original_top_n = candidate_products[:RERANKING_TOP_N]

    candidate_ids = [p.get('id') for p in candidate_products if p.get('id') is not None]
    if not candidate_ids:
         logger.error("Candidate products list is missing IDs.")
         return original_top_n

    prompt = _construct_rerank_prompt(conversation_history, candidate_products)

    logger.info(f"Requesting LLM reranking for {len(candidate_ids)} products.")
    llm_response_content = await generate_text(
        messages=[{"role": "user", "content": prompt}],
        model=RERANKING_MODEL_NAME,
        temperature=0.2, # Lower temperature for more deterministic rating
        max_tokens=800 # Needs enough tokens for JSON output (~15-20 chars per product)
    )

    if llm_response_content is None:
        logger.error("LLM call failed during reranking.")
        return original_top_n

    # Parse and validate the ratings
    ratings = _parse_and_validate_ratings(llm_response_content, candidate_ids)

    if ratings is None:
        logger.warning("Failed to get valid ratings from LLM. Returning original top N.")
        return original_top_n

    # Create a mapping from ID to original product object
    id_to_product = {p["id"]: p for p in candidate_products if "id" in p}

    # Sort products based on the ratings (descending)
    # Tie-breaking: Use original index from candidate_products list
    # This preserves db-ally's relevance order among equally-rated items.
    original_order_map = {pid: i for i, pid in enumerate(candidate_ids)}

    try:
        sorted_product_ids = sorted(
            ratings.keys(),
            # Sort by rating (desc), then by original index (asc)
            key=lambda pid: (-ratings.get(pid, -99), original_order_map.get(pid, 999))
        )
    except Exception as e:
         logger.error(f"Error during sorting based on ratings: {e}", exc_info=True)
         return original_top_n

    # Get the full product objects for the top N
    reranked_products = []
    for pid in sorted_product_ids:
        if pid in id_to_product:
            reranked_products.append(id_to_product[pid])
        else:
            logger.warning(f"Product ID {pid} found in ratings but not in original product map. Skipping.")


    final_top_products = reranked_products[:RERANKING_TOP_N]

    # Log the result for debugging
    final_ids_ratings = {pid: ratings.get(pid) for pid in [p['id'] for p in final_top_products]}
    logger.info(f"Reranking complete. Final Top {RERANKING_TOP_N} IDs & Ratings: {final_ids_ratings}")

    return final_top_products