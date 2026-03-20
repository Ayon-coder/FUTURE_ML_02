import os
import sys
import json
import re
import traceback
import joblib
import numpy as np
import nltk
from scipy.sparse import hstack, csr_matrix
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── API Setup ───────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── NLTK Setup for Vercel ─────────────────────────────────────
NLTK_DATA_DIR = '/tmp/nltk_data'
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

try:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR, quiet=True)
    
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"Warning NLTK: {e}")
    stop_words = set()
    lemmatizer = None

# ── Globals ──────────────────────────────────────────────────
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'backend', 'saved_models')

category_model = None
priority_model = None
category_encoder = None
priority_encoder = None
feature_extractors = None
metrics_data = None
load_error_reason = "Not attempted yet"

def load_models():
    global category_model, priority_model, category_encoder, priority_encoder
    global feature_extractors, metrics_data, load_error_reason
    try:
        category_model = joblib.load(os.path.join(MODELS_DIR, 'category_model.pkl'))
        priority_model = joblib.load(os.path.join(MODELS_DIR, 'priority_model.pkl'))
        category_encoder = joblib.load(os.path.join(MODELS_DIR, 'category_encoder.pkl'))
        priority_encoder = joblib.load(os.path.join(MODELS_DIR, 'priority_encoder.pkl'))
        feature_extractors = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
        
        metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        return True
    except Exception as e:
        load_error_reason = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return False

# Attempt to load immediately
load_models()

# ── Preprocessing Logic ──────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    words = text.split()
    if stop_words:
        words = [w for w in words if w not in stop_words]
    if lemmatizer:
        words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def safe_encode(encoder, value):
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        return 0

# ── Routes ───────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    loaded = category_model is not None and feature_extractors is not None
    return jsonify({
        'status': 'healthy' if loaded else 'models_not_loaded',
        'models_loaded': loaded,
        'message': 'Minimal Vercel Serverless API Running',
        'error': load_error_reason if not loaded else None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if category_model is None or feature_extractors is None:
        if not load_models():
            return jsonify({'error': 'Models not loaded.', 'details': load_error_reason}), 503

    try:
        data = request.json
        subject = str(data.get('subject', '')).strip()
        description = str(data.get('description', '')).strip()
        product = str(data.get('product', 'unknown')).strip()
        channel = str(data.get('channel', 'unknown')).strip()
        age = int(data.get('customer_age', 30))
        
        if not subject and not description:
            return jsonify({'error': 'Please provide subject and/or description.'}), 400

        combined = f"{subject} {description}".strip()
        processed_text = preprocess_text(combined)
        
        # Extract features
        text_vec = feature_extractors['text_vectorizer'].transform([processed_text])
        subj_vec = feature_extractors['subject_vectorizer'].transform([subject])
        
        prod_enc = safe_encode(feature_extractors['product_encoder'], product)
        chan_enc = safe_encode(feature_extractors['channel_encoder'], channel)
        cat_enc = safe_encode(feature_extractors['subject_cat_encoder'], subject)
        age_scaled = feature_extractors['age_scaler'].transform([[age]])[0][0]
        
        struct_sparse = csr_matrix(np.array([[prod_enc, chan_enc, cat_enc, age_scaled]]))
        X = hstack([text_vec, subj_vec, struct_sparse])
        
        # Predict Category
        cat_encoded = category_model.predict(X)
        cat_proba = category_model.predict_proba(X)
        cat_pred = category_encoder.inverse_transform(cat_encoded)
        
        # Predict Priority
        pri_encoded = priority_model.predict(X)
        pri_proba = priority_model.predict_proba(X)
        pri_pred = priority_encoder.inverse_transform(pri_encoded)
        
        # Results
        cat_conf = float(np.max(cat_proba[0]))
        pri_conf = float(np.max(pri_proba[0]))
        
        res = {
            'category': cat_pred[0],
            'category_confidence': round(cat_conf, 4),
            'category_probabilities': {
                cls: round(float(prob), 4) for cls, prob in zip(category_encoder.classes_, cat_proba[0])
            },
            'priority': pri_pred[0],
            'priority_confidence': round(pri_conf, 4),
            'priority_probabilities': {
                cls: round(float(prob), 4) for cls, prob in zip(priority_encoder.classes_, pri_proba[0])
            }
        }
        return jsonify({'success': True, 'predictions': [res]})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}', 'trace': traceback.format_exc()}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if metrics_data is None:
        if not load_models() or metrics_data is None:
            return jsonify({
                'error': 'Metrics not found. Please train models first.',
                'details': load_error_reason
            }), 404
    return jsonify(metrics_data)

# Allow local testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
