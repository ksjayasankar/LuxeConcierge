# luxury_leather_chatbot/core/retrieval_module.py

import logging
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import litellm
import re
import copy # For deepcopy

# --- Project Imports ---
try:
    from config import (
        EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_N, FAISS_INDEX_DIR,
        FAISS_INDEX_FILE, NVIDIA_NIM_API_KEY, NVIDIA_NIM_API_BASE,
        SEMANTIC_FILTER_SIMILARITY_THRESHOLD,
        # Import new relaxation configs
        RELAX_SUSTAINABILITY_STEP, RELAX_PRICE_MAX_PERCENT, MIN_SUSTAINABILITY_FLOOR
    )
    from database import get_session
    from models import Product
except ImportError as e:
     print(f"Error importing project config/db/models in retrieval_module.py: {e}")
     raise

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Globals ---
_faiss_index: Optional[faiss.Index] = None
_product_data_store: List[Dict[str, Any]] = []
_product_id_to_faiss_index: Dict[int, int] = {}
_faiss_index_to_product_id: Dict[int, int] = {}
# Store for pre-computed attribute embeddings
_attribute_embeddings: Dict[Tuple[str, str], np.ndarray] = {}
# Define which attributes use semantic filtering
_ATTRIBUTES_TO_EMBED = ['style', 'color', 'occasion']

# === Helper Functions ===

def format_product_for_embedding(product: Dict[str, Any]) -> str:
    # (Keep this function as it was)
    color_info = str(product.get('color', 'N/A')).replace('/', ', ')
    occasion_info = str(product.get('occasion', 'N/A')).replace(',', ';')
    leather_type_info = str(product.get('leather_type', 'N/A'))
    score_text = f"Quality rating - Craftsmanship: {product.get('craftsmanship_score', 0.0):.1f}/10, Sustainability: {product.get('sustainability_score', 0.0):.1f}/10."
    material_text = f"Material: {leather_type_info}."
    price_text = f"Price category: Around ${product.get('price', 0.0):.0f}."
    representation = (
        f"Name: {product.get('name', '')}. "
        f"Style: {product.get('style', '')}. "
        f"Color: {color_info}. "
        f"{material_text} {score_text} {price_text} "
        f"Ideal for: {occasion_info}. "
        f"Description highlights: {product.get('description', '')[:250]}..."
    )
    return representation.replace('..', '.')

# === Helper for getting embeddings safely ===

# In core/retrieval_module.py

async def _get_embeddings_batch(texts: List[str], input_type: str = "passage") -> List[Optional[np.ndarray]]:
    """Gets dense embeddings for a batch of texts using litellm, handling errors."""
    embeddings_list = []
    if not texts:
        return []
    try:
        # FIX: Removed the extra_body parameter
        response = await litellm.aembedding(
            model=EMBEDDING_MODEL_NAME,
            input=texts,
            api_key=NVIDIA_NIM_API_KEY,
            api_base=NVIDIA_NIM_API_BASE,
            input_type=input_type
        )

        if response.data and len(response.data) == len(texts):
            # FIX: This now maps to a simple np.ndarray, not a tuple
            embedding_map = {item['index']: np.array(item['embedding'], dtype=np.float32) for item in response.data}
            embeddings_list = [embedding_map.get(i) for i in range(len(texts))]

            # Normalize embeddings (this logic is correct and should be kept)
            for i, emb in enumerate(embeddings_list):
                if emb is not None:
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        embeddings_list[i] = emb / norm
                    else:
                        logger.warning(f"Embedding for text '{texts[i]}' has zero norm.")
                        embeddings_list[i] = None
        else:
            logger.error(f"Mismatched response length from embedding API. Expected {len(texts)}, got {len(response.data) if response.data else 0}.")
            embeddings_list = [None] * len(texts)

    except Exception as e:
        # Shortened error handling for brevity, your original was fine
        logger.error(f"Unexpected Error during batch embedding for input_type '{input_type}': {e}", exc_info=True)
        embeddings_list = [None] * len(texts)

    failed_count = sum(1 for emb in embeddings_list if emb is None)
    if failed_count > 0:
        logger.warning(f"Failed to get embeddings for {failed_count}/{len(texts)} texts in the batch.")

    return embeddings_list

# === Initialization Function ===
async def initialize_retrieval_system(force_rebuild: bool = False):
    # (Keep this function as it was, including Step 3 for pre-calculating attribute embeddings)
    """Loads products, generates/loads Faiss index, AND pre-computes attribute embeddings."""
    global _faiss_index, _product_data_store, _attribute_embeddings
    global _product_id_to_faiss_index, _faiss_index_to_product_id

    if not NVIDIA_NIM_API_KEY or not NVIDIA_NIM_API_BASE:
        logger.critical("NVIDIA NIM API Key or Base URL not configured.")
        raise ValueError("Missing NVIDIA NIM API Key/Base URL configuration.")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    # Step 1: Load products from DB
    logger.info("Loading products from SQLite database...")
    session = None
    try:
        session = get_session()
        all_products_db = session.query(Product).order_by(Product.id).all()
        _product_data_store = [p.to_dict() for p in all_products_db]
        _product_id_to_faiss_index = {p['id']: i for i, p in enumerate(_product_data_store)}
        _faiss_index_to_product_id = {i: p['id'] for i, p in enumerate(_product_data_store)}
        logger.info(f"Loaded {len(_product_data_store)} products from database.")
        if not _product_data_store:
             logger.warning("Database contains no products."); _faiss_index = None; return
    except Exception as e:
        logger.critical(f"Failed to load products from database: {e}", exc_info=True)
        raise RuntimeError("Failed to load product data from database.") from e
    finally:
        if session: session.close()

    # Step 2: Load or Build Faiss Index
    index_loaded = False
    if not force_rebuild and os.path.exists(FAISS_INDEX_FILE):
        # ... (Faiss loading logic) ...
        logger.info(f"Attempting to load existing Faiss index from: {FAISS_INDEX_FILE}")
        try:
            loaded_index = faiss.read_index(FAISS_INDEX_FILE)
            if loaded_index.ntotal == len(_product_data_store):
                _faiss_index = loaded_index; index_loaded = True
                logger.info(f"Successfully loaded Faiss index ({_faiss_index.ntotal} vectors).")
            else: logger.warning(f"Index size ({loaded_index.ntotal}) != DB count ({len(_product_data_store)}). Rebuilding.")
        except Exception as e: logger.error(f"Failed to load existing Faiss index: {e}. Rebuilding...", exc_info=True)

    if not index_loaded:
        logger.info("Building new Faiss index from database products (dense vectors only)...")
        product_texts = [format_product_for_embedding(p) for p in _product_data_store]
        batch_size = 16
        all_embeddings_list = []
        for i in range(0, len(product_texts), batch_size):
            batch_texts_segment = product_texts[i:i+batch_size]
            logger.info(f"Embedding product descriptions batch {i//batch_size + 1}/{(len(product_texts)+batch_size-1)//batch_size}...")
            # FIX: This now returns a list of dense vectors
            batch_embeddings = await _get_embeddings_batch(batch_texts_segment, input_type="passage")
            all_embeddings_list.extend(batch_embeddings)

        valid_embeddings = [emb for emb in all_embeddings_list if emb is not None]
        if len(valid_embeddings) != len(_product_data_store):
            logger.critical(f"Failed to generate embeddings for all products ({len(valid_embeddings)}/{len(_product_data_store)} successful). Cannot build index.")
            raise RuntimeError("Failed to generate embeddings for all products.")

        embeddings_np = np.array(valid_embeddings).astype(np.float32)
        logger.info(f"Product Embeddings generated. Shape: {embeddings_np.shape}")
        dimension = embeddings_np.shape[1]
        logger.info(f"Creating Faiss IndexFlatIP dim {dimension}...")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)
        logger.info(f"Faiss index created. Total vectors: {index.ntotal}")

        try:
            logger.info(f"Saving new Faiss index to: {FAISS_INDEX_FILE}")
            faiss.write_index(index, FAISS_INDEX_FILE)
        except Exception as e:
            logger.error(f"Failed to save new Faiss index: {e}", exc_info=True)
        _faiss_index = index

    # Step 3: Pre-calculate Attribute Embeddings
    logger.info("Pre-calculating embeddings for filterable attributes...")
    _attribute_embeddings = {} # Clear previous if any
    unique_attribute_values_map = {}
    texts_to_embed_for_attrs = []
    current_index = 0
    for attr_name in _ATTRIBUTES_TO_EMBED:
        unique_values = set()
        for product in _product_data_store:
            value = product.get(attr_name)
            if value and isinstance(value, str):
                 unique_values.add(value.strip())
        for value in unique_values:
            if value:
                key = (attr_name, value)
                if key not in unique_attribute_values_map:
                    unique_attribute_values_map[key] = current_index
                    texts_to_embed_for_attrs.append(value)
                    current_index += 1

    logger.info(f"Found {len(texts_to_embed_for_attrs)} unique attribute values across {_ATTRIBUTES_TO_EMBED} to embed.")
    if texts_to_embed_for_attrs:
        batch_size = 32
        all_attr_embeddings = []
        for i in range(0, len(texts_to_embed_for_attrs), batch_size):
            batch_texts_segment = texts_to_embed_for_attrs[i:i+batch_size]
            logger.info(f"Embedding attributes batch {i//batch_size + 1}/{(len(texts_to_embed_for_attrs)+batch_size-1)//batch_size}...")
            # FIX: Call the reverted helper, which returns a list of dense vectors
            batch_embeddings = await _get_embeddings_batch(batch_texts_segment, input_type="passage")
            all_attr_embeddings.extend(batch_embeddings)

        total_embedded_count = 0
        for key, index in unique_attribute_values_map.items():
            # FIX: Check the index and get the dense vector directly
            if index < len(all_attr_embeddings) and all_attr_embeddings[index] is not None:
                _attribute_embeddings[key] = all_attr_embeddings[index]
                total_embedded_count += 1
            else:
                 logger.warning(f"Failed to get embedding for attribute {key}. It won't be available for filtering.")
        logger.info(f"Successfully pre-calculated embeddings for {total_embedded_count}/{len(texts_to_embed_for_attrs)} unique attribute values.")
    else:
         logger.info("No unique attribute values found to pre-calculate embeddings for.")

    logger.info("Retrieval system initialization complete.")


def is_retrieval_initialized() -> bool:
    # Check Faiss index and also if attribute embeddings dict is populated (or attempted)
    return _faiss_index is not None and _faiss_index.ntotal > 0 and bool(_product_data_store)


# === Filtering Function (Includes Semantic + Fallbacks) ===
def check_filters_detailed(
    product: Dict[str, Any],
    criteria: Dict[str, Any],
    criteria_embeddings: Dict[str, Optional[np.ndarray]]
) -> tuple[bool, str]:
    # (Keep the last working version of this function, including the fix for the TypeError)
    """
    Checks if a product satisfies the extracted criteria using semantic similarity
    for text fields (defined in _ATTRIBUTES_TO_EMBED) and rules for others.
    Includes fallbacks for exact matches and specific occasion keyword logic.
    Returns (boolean, reason_string).
    Assumes embeddings provided are normalized. Returns tuple(bool, str).
    """
    global _attribute_embeddings # Access pre-calculated product attribute embeddings
    global _ATTRIBUTES_TO_EMBED # Access list of attributes to check semantically

    # --- Check 0: No Criteria ---
    if not criteria: return True, "No criteria"

    # --- Check 1: Price (Rule-based) ---
    try:
        product_price = float(product.get('price', 0))
        if criteria.get('price_min') is not None and product_price < float(criteria['price_min']):
            return False, f"Price {product_price} < min {criteria['price_min']}"
        if criteria.get('price_max') is not None and product_price > float(criteria['price_max']):
            return False, f"Price {product_price} > max {criteria['price_max']}"
    except (ValueError, TypeError): pass

    # --- Check 2: Scores (Rule-based) ---
    try:
        if criteria.get('min_craftsmanship') is not None and product.get('craftsmanship_score', 0.0) < float(criteria['min_craftsmanship']):
             return False, f"Craftsmanship {product.get('craftsmanship_score', 0.0)} < min {criteria['min_craftsmanship']}"
        if criteria.get('min_sustainability') is not None and product.get('sustainability_score', 0.0) < float(criteria['min_sustainability']):
             return False, f"Sustainability {product.get('sustainability_score', 0.0)} < min {criteria['min_sustainability']}"
    except (ValueError, TypeError): pass

    # --- Check 3: Material Preference (Rule-based) ---
    if criteria.get('material_preference') and criteria['material_preference'] != 'any':
        preference = criteria['material_preference'].lower()
        product_leather = product.get('leather_type', '').lower()
        real_leathers = ['full-grain', 'top-grain']; vegan_leathers = ['vegan leather']; synthetics = ['leatherette', 'polyester']
        is_real = any(rl in product_leather for rl in real_leathers)
        is_vegan = any(vl in product_leather for vl in vegan_leathers)
        is_synthetic = any(syn in product_leather for syn in synthetics)
        if preference == 'real_leather' and not is_real: return False, f"Material pref 'real_leather' failed for type '{product_leather}'"
        if preference == 'vegan' and not is_vegan: return False, f"Material pref 'vegan' failed for type '{product_leather}'"
        if preference == 'synthetic' and not is_synthetic: return False, f"Material pref 'synthetic' failed for type '{product_leather}'"

    # --- Check 4: Semantic Attributes (Style, Color, Occasion) ---
    for attr_name in _ATTRIBUTES_TO_EMBED:
        criteria_value = criteria.get(attr_name)
        if criteria_value and isinstance(criteria_value, str):
            product_value = product.get(attr_name)
            if not product_value or not isinstance(product_value, str):
                 logger.debug(f"Product ID {product.get('id', 'N/A')} missing required attribute '{attr_name}' for criteria '{criteria_value}'")
                 return False, f"Filter fail: Product missing required attribute '{attr_name}'"

            criteria_emb = criteria_embeddings.get(attr_name)
            product_emb_key = (attr_name, product_value.strip())
            product_emb = _attribute_embeddings.get(product_emb_key)

            if criteria_emb is None:
                 logger.warning(f"Missing criteria embedding for {attr_name}='{criteria_value}'. Cannot perform semantic check for Product ID {product.get('id', 'N/A')}.")
                 return False, f"Semantic filter fail: Missing embedding for criteria {attr_name}='{criteria_value}'"
            if product_emb is None:
                 logger.warning(f"Missing product embedding for {product_emb_key}. Cannot perform semantic check for Product ID {product.get('id', 'N/A')}.")
                 return False, f"Semantic filter fail: Missing embedding for product {attr_name}='{product_value}'"

            if not isinstance(criteria_emb, np.ndarray) or not isinstance(product_emb, np.ndarray) or criteria_emb.ndim != 1 or product_emb.ndim != 1:
                 logger.error(f"Invalid embedding types or dimensions for {attr_name} ('{criteria_value}' vs '{product_value}'). Shapes: {getattr(criteria_emb, 'shape', 'N/A')}, {getattr(product_emb, 'shape', 'N/A')}")
                 return False, f"Semantic filter error: Invalid embedding structure for {attr_name}"

            try:
                similarity = np.dot(np.ascontiguousarray(criteria_emb, dtype=np.float32), np.ascontiguousarray(product_emb, dtype=np.float32))
            except ValueError as e:
                 logger.error(f"Dot product failed for {attr_name} ('{criteria_value}' vs '{product_value}'). Shapes: {criteria_emb.shape}, {product_emb.shape}. Error: {e}")
                 return False, f"Semantic filter error: Dot product failed for {attr_name}"

            if similarity < SEMANTIC_FILTER_SIMILARITY_THRESHOLD:
                if criteria_value.strip().lower() == product_value.strip().lower():
                    logger.debug(f"Product ID {product.get('id', 'N/A')}: Semantic filter for {attr_name} ({similarity:.3f} < {SEMANTIC_FILTER_SIMILARITY_THRESHOLD}) - PASSING due to exact match.")
                    continue
                elif attr_name == 'occasion':
                     try:
                         product_keywords = set(filter(None, re.split(r'[^a-z0-9]+', product_value.lower())))
                         criteria_keywords = set(filter(None, re.split(r'[^a-z0-9]+', criteria_value.lower())))
                         if criteria_keywords.intersection(product_keywords):
                             logger.debug(f"Product ID {product.get('id', 'N/A')}: Semantic filter for occasion failed (sim {similarity:.3f}), but PASSING due to keyword overlap ({criteria_keywords.intersection(product_keywords)}).")
                             continue
                         elif ('travel' in product_keywords) and ('weekend' in criteria_keywords or 'getaway' in criteria_keywords):
                             logger.debug(f"Product ID {product.get('id', 'N/A')}: Semantic filter for occasion failed (sim {similarity:.3f}), but PASSING due to related 'weekend/getaway' vs 'travel' keywords.")
                             continue
                     except Exception as e:
                         logger.error(f"Error during occasion keyword fallback check for Product ID {product.get('id', 'N/A')}, Attr: {attr_name}, Values: '{criteria_value}' vs '{product_value}'. Error: {e}")
                         return False, f"Error during occasion fallback check for {attr_name}" # Fail safely

                return False, (f"Semantic filter fail: {attr_name} '{product_value}' "
                               f"(sim: {similarity:.3f}) below threshold {SEMANTIC_FILTER_SIMILARITY_THRESHOLD} "
                               f"for criteria '{criteria_value}'")
    # If loop completes
    return True, "Passed all filters"


# === Core Retrieval Function (Includes Relaxation Logic) ===
# In core/retrieval_module.py

async def get_product_candidates_hybrid(
    natural_language_query: str,
    criteria: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Retrieves product candidates using Faiss (dense search), followed by
    filtering, relaxation, and metadata-based score boosting.
    """
    global _faiss_index, _product_data_store, _faiss_index_to_product_id

    if not is_retrieval_initialized():
        logger.error("Retrieval system not initialized.")
        return [], None

    logger.info(f"Retrieving candidates for query: '{natural_language_query}' with STRICT criteria: {criteria}")

    # Step 1: Embed Query and Search Faiss (Dense Search)
    try:
        query_embed_list = await _get_embeddings_batch([natural_language_query], input_type="query")
        if not (query_embed_list and query_embed_list[0] is not None):
            raise ValueError("Query embedding failed.")
        query_embedding_np = query_embed_list[0].reshape(1, -1)

        M = min(_faiss_index.ntotal, 75)
        if M == 0: return [], None

        logger.debug(f"Searching Faiss index for top {M} candidates...")
        distances, indices = _faiss_index.search(query_embedding_np, M)
        top_m_indices = indices[0][indices[0] != -1]
        top_m_scores = distances[0][indices[0] != -1]
        logger.debug(f"Faiss search returned {len(top_m_indices)} valid candidates.")
    except Exception as e:
        logger.error(f"Error during query embedding or Faiss search: {e}", exc_info=True)
        return [], None

    # Step 1.5: Embed CRITERIA values (Dense Vectors Only)
    criteria_embeddings_vectors = {}
    criteria_texts_to_embed_list = [v for k, v in criteria.items() if k in _ATTRIBUTES_TO_EMBED and isinstance(v, str)]
    if criteria_texts_to_embed_list:
        criteria_embeddings = await _get_embeddings_batch(criteria_texts_to_embed_list, input_type="passage")
        # Map back to criteria keys
        text_to_emb = dict(zip(criteria_texts_to_embed_list, criteria_embeddings))
        for key in _ATTRIBUTES_TO_EMBED:
            if key in criteria:
                criteria_embeddings_vectors[key] = text_to_emb.get(criteria[key])


    # Step 2: Filter and Relax Logic
    def _filter_candidates(indices_to_check, scores_to_check, current_criteria):
        passed_candidates = []
        for faiss_idx, score in zip(indices_to_check, scores_to_check):
            product = _product_data_store[int(faiss_idx)]
            passes, reason = check_filters_detailed(product, current_criteria, criteria_embeddings_vectors)
            if passes:
                passed_candidates.append({"product": product, "semantic_score": float(score)})
        return passed_candidates

    # Strict filtering attempt
    final_candidates_with_scores = _filter_candidates(top_m_indices, top_m_scores, criteria)
    relaxation_applied = None

    # Relaxation Logic (works as is)
    if not final_candidates_with_scores and len(top_m_indices) > 0:
        logger.info("Strict filtering yielded 0 results. Attempting relaxation...")
        relaxation_steps = [
            {"name": "sustainability score", "param": "min_sustainability", "type": "lower_step", "step": RELAX_SUSTAINABILITY_STEP, "floor": MIN_SUSTAINABILITY_FLOOR},
            {"name": "maximum price", "param": "price_max", "type": "increase_percent", "percent": RELAX_PRICE_MAX_PERCENT},
        ]
        for step_info in relaxation_steps:
            param, original_value = step_info["param"], criteria.get(step_info["param"])
            if original_value is not None:
                relaxed_criteria = copy.deepcopy(criteria)
                # (Your logic to calculate new_value)
                new_value = None
                if step_info["type"] == "lower_step":
                    new_value = float(original_value) - step_info["step"]
                    if "floor" in step_info: new_value = max(new_value, step_info["floor"])
                elif step_info["type"] == "increase_percent":
                    new_value = float(original_value) * (1.0 + step_info["percent"])

                if new_value is not None and abs(float(original_value) - new_value) > 1e-6:
                    relaxed_criteria[param] = new_value
                    logger.info(f"Relaxation Attempt: Relaxing {step_info['name']} to {new_value:.2f}")
                    relaxed_results = _filter_candidates(top_m_indices, top_m_scores, relaxed_criteria)
                    if relaxed_results:
                        logger.info(f"Found {len(relaxed_results)} candidates after relaxing {step_info['name']}.")
                        final_candidates_with_scores = relaxed_results
                        relaxation_applied = step_info['name']
                        break
            if final_candidates_with_scores: break

    # Step 3: Boost, Rank, and Return
    if not final_candidates_with_scores:
        logger.info("No candidates found even after relaxation attempts.")
        return [], None

    # Re-introduce boosting
    logger.info(f"Applying boosting and ranking to {len(final_candidates_with_scores)} candidates.")
    w_semantic = 1.0
    w_craftsmanship = 0.15 # Slightly increased weight
    w_sustainability = 0.05
    for candidate in final_candidates_with_scores:
        product = candidate["product"]
        craft_score = product.get('craftsmanship_score', 5.0)
        sustain_score = product.get('sustainability_score', 5.0)
        boosted_score = (w_semantic * candidate["semantic_score"] +
                         w_craftsmanship * (craft_score / 10.0) + # Normalize to 0-1 range
                         w_sustainability * (sustain_score / 10.0))
        candidate["final_score"] = boosted_score

    ranked_candidates = sorted(final_candidates_with_scores, key=lambda x: x["final_score"], reverse=True)

    # Return product dictionaries, sliced to the configured limit for the reranker
    final_results = [c["product"] for c in ranked_candidates[:RETRIEVAL_TOP_N]]
    logger.info(f"Returning top {len(final_results)} candidates for reranking.")

    return final_results, relaxation_applied