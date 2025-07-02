# luxury_leather_chatbot/config.py
import os
from dotenv import load_dotenv
import logging

load_dotenv()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = "luxury_leather_chatbot"

# --- Database ---
DB_NAME = "luxury_leather.db"
DB_URL = f"sqlite:///{os.path.join(BASE_DIR, DB_NAME)}"

# --- LLM Configuration (DeepSeek) ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
CONVERSATIONAL_MODEL_NAME = "deepseek-chat" # Used for conversation logic
RERANKING_MODEL_NAME = "deepseek-chat"    # Used for reranking candidates

# --- User Profile Settings ---
USER_PROFILES_DIR = os.path.join(BASE_DIR, "user_profiles")
USER_PROFILE_RETENTION_DAYS = 90  # How long to keep inactive profiles
USER_PROFILE_MAX_INTERACTIONS = 100  # Maximum interactions to store per user
USER_PROFILE_MIN_INTERACTIONS_FOR_LEARNING = 3  # Minimum interactions before using profile for personalization

# --- Profile Learning Thresholds ---
PROFILE_PRICE_BUFFER_PERCENT = 0.15  # 15% buffer when using historical price preferences
PROFILE_QUALITY_BUFFER = 1.0  # Reduce quality requirements by 1 point from average
PROFILE_SUSTAINABILITY_THRESHOLD = 7.0  # Min avg sustainability score to consider user sustainability-focused

# --- NVIDIA NIM Embeddings (Used directly via litellm) ---
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
NVIDIA_NIM_API_BASE = os.getenv("NVIDIA_NIM_EMBEDDING_BASE") # Base URL for embeddings endpoint
# Ensure this matches the model string used in litellm calls
EMBEDDING_MODEL_NAME = "nvidia_nim/baai/bge-m3"

# --- Faiss ---
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_indexes")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "products.index")
# No longer need PRODUCT_DATA_STORE_FILE as data is loaded fresh from DB

# --- Flask ---
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-this-in-production-fallback-key")

# --- Application Settings ---
RETRIEVAL_TOP_N = 30 # Number of candidates after retrieval + filtering + boosting
RERANKING_TOP_N = 3  # Number of candidates after LLM reranking
SEMANTIC_FILTER_SIMILARITY_THRESHOLD = 0.5

# --- Relaxation Settings ---
RELAX_SUSTAINABILITY_STEP = 0.5  # How much to lower min_sustainability by
RELAX_PRICE_MAX_PERCENT = 0.15 # Increase max price by 15%
# RELAX_PRICE_MIN_PERCENT = 0.15 # Decrease min price by 15% (Can add later)
# RELAX_SIMILARITY_STEP = 0.05 # How much to lower similarity threshold by (Can add later)
MIN_SUSTAINABILITY_FLOOR = 5.0 # Don't relax sustainability below this

# --- Logging ---
LOGGING_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Error Handling ---
FALLBACK_ERROR_MESSAGE = "Sorry, I encountered an issue and couldn't process your request. Please try again."

# --- Check Critical Configs ---
if not DEEPSEEK_API_KEY:
    print("WARNING: DEEPSEEK_API_KEY environment variable not set. LLM calls will fail.")
if not NVIDIA_NIM_API_KEY:
    print("WARNING: NVIDIA_NIM_API_KEY environment variable not set. Embeddings will fail.")
if not NVIDIA_NIM_API_BASE:
    print("WARNING: NVIDIA_NIM_EMBEDDING_BASE environment variable not set. Embeddings will fail.")