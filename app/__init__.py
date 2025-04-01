from flask import Flask
from flask_cors import CORS
import logging
import os

def create_app():
    app = Flask(__name__)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable CORS for all routes with more specific settings
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Import and register blueprints/routes
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    # Configure app-specific logging
    logger = logging.getLogger(__name__)
    logger.info("Guitar Tabs API initialized")
    
    # Log environment status
    try:
        # Try importing Spleeter to check if it's available
        from .services.source_separation import SPLEETER_AVAILABLE
        logger.info(f"Spleeter available: {SPLEETER_AVAILABLE}")
    except ImportError:
        logger.warning("Spleeter import check failed")
    
    return app 