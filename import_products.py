# luxury_leather_chatbot/import_products.py

import os
import sys # Keep sys only for exit, not path manipulation
import json
import logging
from sqlalchemy.exc import SQLAlchemyError
from flask import Flask # Keep this standard import

# --- Direct Imports from Files in the SAME Directory ---
try:
    # NO leading dots, NO package prefix
    from database import db, init_db
    from models import Product
    from config import DB_URL, FLASK_SECRET_KEY
except ImportError as e:
    print(f"ERROR: Failed to import required modules ({e}).")
    print("Please ensure:")
    print("  1. You are running this script using 'python3 import_products.py'")
    print("     from WITHIN the 'luxury-leather-chatbot' directory.")
    print("  2. The files database.py, models.py, config.py exist in this directory.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Function import_products_from_json (Keep content as before) ---
def import_products_from_json(app, json_file_path):
    # ... (function content remains the same - it uses the passed 'app')
    if not os.path.exists(json_file_path):
        logger.error(f"JSON file not found at {json_file_path}")
        return
    with app.app_context():
        try:
            if db.session.query(Product.id).count() > 0:
                logger.info("Products table appears to contain data. Skipping import.")
                return
        except Exception as e:
            logger.warning(f"Could not check for existing products (maybe table doesn't exist?): {e}")
        logger.info(f"Importing products from {json_file_path}...")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                products_data = json.load(f)
        except json.JSONDecodeError as e: logger.error(f"Error decoding JSON file: {e}"); return
        except Exception as e: logger.error(f"Error reading JSON file: {e}"); return
        products_added = 0; products_skipped = 0
        for product_data in products_data:
            if not all(k in product_data for k in ['name', 'price', 'leather_type', 'color', 'style']):
                 logger.warning(f"Skipping product (missing fields): {product_data.get('name', 'N/A')}"); products_skipped += 1; continue
            try:
                price = float(product_data['price'])
                craftsmanship = product_data.get('craftsmanship_score'); sustainability = product_data.get('sustainability_score')
                product = Product(name=product_data['name'], description=product_data.get('description', ''), price=price, leather_type=product_data['leather_type'], color=product_data['color'], style=product_data['style'], hardware=product_data.get('hardware'), occasion=product_data.get('occasion'), craftsmanship_score=float(craftsmanship) if craftsmanship is not None else None, sustainability_score=float(sustainability) if sustainability is not None else None)
                db.session.add(product); products_added += 1
            except ValueError as e: logger.warning(f"Skipping '{product_data.get('name', 'N/A')}' (value error): {e}"); products_skipped += 1
            except Exception as e: logger.error(f"Skipping '{product_data.get('name', 'N/A')}' (creation error): {e}"); products_skipped += 1
        if products_added > 0:
            logger.info(f"Attempting to commit {products_added} products...");
            try: db.session.commit(); logger.info(f"Successfully added {products_added} products.")
            except Exception as e: db.session.rollback(); logger.error(f"Commit failed: {e}")
        else: logger.info("No new valid products were processed.")
        if products_skipped > 0: logger.warning(f"Skipped {products_skipped} products.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Running Product Import Script ---")
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = FLASK_SECRET_KEY
    if not FLASK_SECRET_KEY or FLASK_SECRET_KEY.startswith("fallback"):
        print("WARNING: Using default/insecure FLASK_SECRET_KEY.")
    try:
        print(f"Initializing database connection to: {DB_URL}")
        init_db(app)
        print("Database initialized.")
    except Exception as e: print(f"CRITICAL: Failed to initialize database: {e}"); exit(1)
    try:
        with app.app_context(): print("Ensuring database tables exist..."); db.create_all(); print("Database tables checked/created.")
    except Exception as e: print(f"CRITICAL: Failed to create database tables: {e}"); exit(1)
    json_path = os.path.join(os.path.dirname(__file__), 'products.json')
    try:
        print(f"Starting import from: {json_path}")
        import_products_from_json(app, json_path)
    except Exception as e: print(f"CRITICAL: Import process failed during execution: {e}")
    print("--- Import process finished ---")