"""
Web Interface Module
This module provides Flask web application interface.
"""

from flask import Flask

def create_app():
    """
    Application factory pattern
    Creates and configures the Flask application
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Register blueprints
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app

__all__ = ['create_app']

