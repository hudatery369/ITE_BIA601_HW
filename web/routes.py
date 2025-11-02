"""
Flask Routes - Simplified Version
"""

from flask import Blueprint, render_template, request, jsonify
import os
import sys
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from genetic_algorithm.ga_engine import GeneticAlgorithm
from data_processing.loader import DataLoader
from data_processing.preprocessor import DataPreprocessor

main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns.tolist()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/api/run-ga', methods=['POST'])
def run_ga():
    try:
        data = request.get_json()
        
        filepath = data.get('filepath')
        target_column = data.get('target_column', 'target')
        population_size = data.get('population_size', 30)
        generations = data.get('generations', 50)
        crossover_rate = data.get('crossover_rate', 0.8)
        mutation_rate = data.get('mutation_rate', 0.1)
        
        if not filepath:
            return jsonify({'error': 'Filepath required'}), 400
        
        if not os.path.isabs(filepath):
            filepath = os.path.join(project_root, filepath)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load and preprocess
        loader = DataLoader()
        df = loader.load_file(filepath)
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found'}), 400
        
        # Check if target column has too many unique values (likely continuous)
        n_unique = df[target_column].nunique()
        n_samples = len(df)
        if n_unique > 20 or n_unique == n_samples:
            return jsonify({
                'error': f'العمود "{target_column}" يبدو أنه رقمي متصل (continuous) وليس فئوي. الرجاء اختيار عمود فئوي مثل "target" أو "class".'
            }), 400
        
        preprocessor = DataPreprocessor()
        X_normalized, y, feature_names = preprocessor.full_preprocessing(df, target_column)
        
        # Run GA
        ga = GeneticAlgorithm(
            population_size=population_size,
            n_generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            alpha=0.9,
            verbose=False
        )
        
        best = ga.fit(X_normalized, y)
        
        selected_indices = ga.get_selected_features().tolist()
        selected_names = [feature_names[i] for i in selected_indices]
        
        return jsonify({
            'success': True,
            'selected_features': selected_names,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'n_total': len(feature_names),
            'fitness': float(best.fitness),
            'accuracy': float(best.accuracy),
            'reduction_percent': float((1 - len(selected_indices) / len(feature_names)) * 100)
        }), 200
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
