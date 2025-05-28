# luxury_leather_chatbot/app.py

import logging
import os
import asyncio
from flask import Flask, send_from_directory, render_template, session

# --- Absolute Imports ---
from config import (
    FLASK_SECRET_KEY, LOGGING_LEVEL, LOG_FORMAT,
    PACKAGE_NAME
)
from database import init_db, db, get_session # Import get_session if needed elsewhere
from api.chat_routes import bp as chat_bp
# *** Import the NEW retrieval functions ***
from core.retrieval_module import (
    initialize_retrieval_system,
    is_retrieval_initialized
)
# Removed db-ally specific imports

logging.basicConfig(level=LOGGING_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
# No need for _db_ally_ready flag anymore

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(PACKAGE_NAME, instance_relative_config=False)
    app.config['SECRET_KEY'] = FLASK_SECRET_KEY
    app.config.from_object('config') # Load config from config.py

    logger.info("Initializing Database...")
    try:
        db_engine = init_db(app)
        # --- Database Table Creation ---
        # Ensure tables are created within app context after DB init
        with app.app_context():
             db.create_all()
             logger.info("Database tables checked/created.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize database or create tables: {e}", exc_info=True)
        # Optionally, prevent app startup if DB fails
        raise RuntimeError("Database initialization failed.") from e


    # --- *** Initialize the NEW Retrieval System (Faiss Index) *** ---
    # Needs to run within an event loop because initialize_retrieval_system is async
    logger.info("Attempting to initialize Retrieval System (Faiss index & embeddings)...")
    try:
        # Using asyncio.run() is simple but can sometimes conflict with Flask's own loop
        # or run twice with the reloader.
        # Alternatives: Use Flask's @app.before_first_request (deprecated) or integrate with
        # a proper async framework startup event if using one (e.g., Quart, FastAPI style).
        # For standard Flask + dev server, asyncio.run might be sufficient for startup.
        asyncio.run(initialize_retrieval_system()) # Pass force_rebuild=True for development if needed

        if is_retrieval_initialized():
             logger.info("Retrieval System initialized successfully.")
        else:
             logger.warning("Retrieval System initialization finished, but it seems not ready (check logs).")

    except RuntimeError as e:
         logger.error(f"RuntimeError during retrieval system initialization (maybe loop issue?): {e}", exc_info=True)
         # Decide if this is fatal for the app
    except FileNotFoundError as e:
         logger.critical(f"FATAL: Required file not found during retrieval init: {e}")
         raise # Usually fatal if product data/index cannot be built/loaded
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize Retrieval System: {e}", exc_info=True)
        # Decide if app should start without retrieval. Probably not.
        raise RuntimeError("Retrieval system initialization failed.") from e
    # --- End Retrieval System Initialization ---


    logger.info("Registering API blueprints...")
    app.register_blueprint(chat_bp, url_prefix='/api')
    logger.info("API blueprint registered.")

    # --- Basic Routes ---
    @app.route('/')
    def customer_interface():
        """Serves the main customer chat page."""
        session.permanent = True
        return render_template('customer/index.html')

    @app.route('/admin')
    def admin_dashboard():
        # Add authentication/authorization for admin routes
        return render_template('admin/index.html')

    # Serve Static files (Development)
    @app.route('/static/<path:path>')
    def send_static(path):
        # Consider using Whitenoise or configuring webserver (Nginx/Apache) for production
        return send_from_directory('static', path)

    logger.info("Flask application created successfully.")
    return app

# --- Main Execution ---
if __name__ == '__main__':
    flask_app = create_app()
    logger.info("Starting Flask development server...")
    # use_reloader=True can cause initialization (like asyncio.run) to run twice.
    # Set use_reloader=False if you encounter issues with double initialization.
    flask_app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=True)