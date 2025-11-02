"""
تطبيق Flask لاختيار الميزات بالخوارزمية الجينية
Flask Application for Genetic Feature Selection
"""
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from algorithms import GeneticFeatureSelection, TraditionalFeatureSelection
import json
import queue

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# إنشاء مجلد الرفع إذا لم يكن موجوداً
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

# مساحة تخزين للتحديثات في الوقت الفعلي
progress_queues = {}

def allowed_file(filename):
    """التحقق من امتداد الملف"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/progress/<session_id>')
def progress(session_id):
    """إرجاع التحديثات في الوقت الفعلي باستخدام Server-Sent Events"""
    def generate():
        q = queue.Queue()
        progress_queues[session_id] = q
        
        try:
            while True:
                data = q.get(timeout=30)
                if data is None:  # إشارة انتهاء
                    break
                yield f"data: {json.dumps(data)}\n\n"
        except queue.Empty:
            pass
        finally:
            progress_queues.pop(session_id, None)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """معالجة رفع الملف وتشغيل الخوارزميات"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'يجب أن يكون الملف بصيغة CSV'}), 400
        
        # حفظ الملف
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # قراءة البيانات
        df = pd.read_csv(filepath)
        
        # الحصول على عمود الهدف من الطلب
        target_column = request.form.get('target_column')
        if not target_column or target_column not in df.columns:
            target_column = df.columns[-1]  # استخدام آخر عمود كافتراضي
        
        # تحضير البيانات
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = df[target_column]
        
        # معالجة القيم المفقودة
        X = X.fillna(X.mean())
        
        # تحويل الهدف إلى رقمي إن لزم
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # الحصول على الإعدادات من الطلب
        population_size = int(request.form.get('population_size', 50))
        generations = int(request.form.get('generations', 30))
        mutation_rate = float(request.form.get('mutation_rate', 0.1))
        
        # الحصول على session_id للبث المباشر
        session_id = request.form.get('session_id')
        
        # إنشاء callback للبث في الوقت الفعلي
        def progress_callback(data):
            if session_id and session_id in progress_queues:
                progress_queues[session_id].put(data)
        
        # تشغيل الخوارزمية الجينية مع callback
        ga = GeneticFeatureSelection(
            X.values, y.values,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            callback=progress_callback
        )
        genetic_features, genetic_score, history = ga.evolve()
        
        # إرسال إشارة انتهاء
        if session_id and session_id in progress_queues:
            progress_queues[session_id].put({'type': 'completed', 'status': 'genetic_done'})
        
        # مقارنة مع الطرق التقليدية
        if session_id and session_id in progress_queues:
            progress_queues[session_id].put({'type': 'status', 'message': 'جاري المقارنة مع الطرق التقليدية...'})
        
        traditional = TraditionalFeatureSelection(X.values, y.values)
        comparison = traditional.compare_all(genetic_features, genetic_score)
        
        # إعداد النتائج
        results = {
            'success': True,
            'data_info': {
                'n_rows': int(df.shape[0]),
                'n_features': int(X.shape[1]),
                'n_selected': len(genetic_features)
            },
            'genetic_algorithm': {
                'accuracy': float(genetic_score),
                'n_features': len(genetic_features),
                'selected_features': genetic_features,
                'feature_names': [X.columns[i] for i in genetic_features],
                'history': history
            },
            'comparison': {
                'methods': ['الخوارزمية الجينية', 'F-Test', 'Mutual Information', 'RFE', 'Model-Based'],
                'accuracies': [
                    float(comparison['genetic']['accuracy']),
                    float(comparison['f_test']['accuracy']),
                    float(comparison['mutual_info']['accuracy']),
                    float(comparison['rfe']['accuracy']),
                    float(comparison['model_based']['accuracy'])
                ],
                'n_features': [
                    comparison['genetic']['n_features'],
                    comparison['f_test']['n_features'],
                    comparison['mutual_info']['n_features'],
                    comparison['rfe']['n_features'],
                    comparison['model_based']['n_features']
                ]
            }
        }
        
        # حذف الملف بعد المعالجة
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'خطأ في المعالجة: {str(e)}'}), 500

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """الحصول على أسماء الأعمدة من الملف"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'يجب أن يكون الملف بصيغة CSV'}), 400
        
        # قراءة الأعمدة فقط
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        
        return jsonify({'columns': columns, 'n_rows': int(df.shape[0]), 'n_cols': int(df.shape[1])})
    
    except Exception as e:
        return jsonify({'error': f'خطأ في قراءة الملف: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
