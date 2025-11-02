"""
WSGI Configuration for PythonAnywhere
This file tells PythonAnywhere how to run your Flask application.
Project: Genetic Feature Selection
User: RawanTalal
"""

import sys
import os

# Add project root to path
# Adjust this path based on where you upload your project
project_root = '/home/RawanTalal/genetic-feature-selection'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs(os.path.join(project_root, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'results/experiments'), exist_ok=True)

# Import Flask app
from app import app as application

# This line makes WSGI happy
if __name__ == "__main__":
    application.run()

