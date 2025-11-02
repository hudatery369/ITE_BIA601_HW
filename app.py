"""
Genetic Algorithm Feature Selection - Main Application
"""

import os
from web import create_app

app = create_app()

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    
    print("Starting server...")
    print("Open: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
