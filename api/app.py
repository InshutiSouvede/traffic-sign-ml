"""
Flask API for traffic sign classification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.prediction import TrafficSignPredictor
from src.model import TrafficSignCNN
from src.preprocessing import TrafficSignPreprocessor
import json
import time

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads/'
RETRAIN_FOLDER = 'retrain_data/'
MODEL_PATH = 'models/best_model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_FOLDER, exist_ok=True)

# Load class names
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Initialize predictor
predictor = TrafficSignPredictor(MODEL_PATH, class_names)

# Model metrics
model_metrics = {
    'uptime_start': time.time(),
    'total_predictions': 0,
    'average_latency': 0,
    'status': 'online'
}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = time.time() - model_metrics['uptime_start']
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'model_loaded': True
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Predict
        result = predictor.predict(filepath)

        # Update metrics
        model_metrics['total_predictions'] += 1

        # Cleanup
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload data for retraining"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    uploaded_files = []

    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(RETRAIN_FOLDER, filename)
            file.save(filepath)
            uploaded_files.append(filename)

    return jsonify({
        'message': f'Uploaded {len(uploaded_files)} files',
        'files': uploaded_files
    })


@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining"""
    try:
        # Load new data
        preprocessor = TrafficSignPreprocessor()
        X_new, y_new = preprocessor.preprocess_dataset(RETRAIN_FOLDER)

        # Load existing model
        model = TrafficSignCNN(num_classes=len(class_names))
        model.model = keras.models.load_model(MODEL_PATH)

        # Retrain (fine-tune)
        history = model.model.fit(
            X_new, y_new,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Save updated model
        model.model.save(MODEL_PATH)

        # Update predictor
        global predictor
        predictor = TrafficSignPredictor(MODEL_PATH, class_names)

        return jsonify({
            'message': 'Model retrained successfully',
            'final_accuracy': float(history.history['accuracy'][-1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model metrics"""
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)

    metrics['total_predictions'] = model_metrics['total_predictions']
    metrics['uptime_hours'] = (time.time() - model_metrics['uptime_start']) / 3600

    return jsonify(metrics)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)