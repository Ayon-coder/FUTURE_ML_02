"""
Flask REST API
==============
Serves ML predictions and model evaluation metrics.

Endpoints:
    POST /predict    — Classify a single ticket (category + priority + confidence)
    GET  /metrics    — Return stored model evaluation metrics
    GET  /health     — Health check
"""

import os
import sys
import json

# Add backend root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocessing.text_preprocessor import preprocess_text, combine_text_fields
from features.feature_extractor import FeatureExtractor
from models.train_model import TicketClassifierTrainer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ── Load models on startup ──────────────────────────────────
SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved_models')
SAVED_MODELS_DIR = os.path.abspath(SAVED_MODELS_DIR)

trainer = None
feature_extractor = None
metrics_data = None


def load_models():
    """Load trained models and vectorizer."""
    global trainer, feature_extractor, metrics_data
    
    try:
        # Load classifier models
        trainer = TicketClassifierTrainer()
        trainer.load_models(SAVED_MODELS_DIR)
        
        # Load TF-IDF + structured feature extractor
        feature_extractor = FeatureExtractor()
        feature_extractor.load(os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.pkl'))
        
        # Load metrics
        metrics_path = os.path.join(SAVED_MODELS_DIR, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        
        print("✅ Models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Failed to load models: {e}")
        print("   Run 'python main.py' first to train models.")
        return False


# ── API Routes ───────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    models_loaded = trainer is not None and feature_extractor is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'models_not_loaded',
        'models_loaded': models_loaded,
        'message': 'Support Ticket Classifier API is running'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Classify a support ticket.
    
    Request JSON:
        {
            "subject": "Payment issue",
            "description": "I was charged twice for my order..."
        }
    
    Response JSON:
        {
            "category": "Billing inquiry",
            "category_confidence": 0.87,
            "priority": "High",
            "priority_confidence": 0.72
        }
    """
    if trainer is None or feature_extractor is None:
        return jsonify({'error': 'Models not loaded. Train models first.'}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400
    
    subject = data.get('subject', '')
    description = data.get('description', '')
    
    if not subject and not description:
        return jsonify({'error': 'At least one of "subject" or "description" is required'}), 400
    
    try:
        # Preprocess
        combined = combine_text_fields(subject, description)
        processed = preprocess_text(combined)
        
        # Extract features (text + structured)
        X = feature_extractor.transform_single(
            processed_text=processed,
            subject=subject,
            product="unknown",
            channel="unknown",
            age=30
        )
        
        # Predict
        results = trainer.predict(X)
        prediction = results[0]
        
        return jsonify({
            'success': True,
            'input': {
                'subject': subject,
                'description': description
            },
            'prediction': prediction
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model evaluation metrics."""
    if metrics_data is None:
        return jsonify({'error': 'Metrics not available. Train models first.'}), 503
    
    return jsonify({
        'success': True,
        'metrics': metrics_data
    })


# ── Main ─────────────────────────────────────────────────────

if __name__ == '__main__':
    load_models()
    print("\n🚀 Starting Support Ticket Classifier API...")
    print("   → http://localhost:5000")
    print("   → POST /predict   — Classify a ticket")
    print("   → GET  /metrics   — Model performance metrics")
    print("   → GET  /health    — Health check\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
