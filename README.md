# 🎫 TicketIQ — ML-Based Support Ticket Classifier & Prioritizer

An end-to-end machine learning system that **automatically classifies customer support tickets** into categories and **assigns priority levels** — helping businesses respond faster, reduce backlog, and improve customer satisfaction.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask)

---

## 📁 Project Structure

```
FUTURE_ML_02/
├── backend/                        # ML Pipeline & API
│   ├── preprocessing/
│   │   └── text_preprocessor.py    # Text cleaning, stopwords, lemmatization
│   ├── features/
│   │   └── feature_extractor.py    # TF-IDF + structured feature extraction
│   ├── models/
│   │   ├── train_model.py          # LinearSVC (category) + RandomForest (priority)
│   │   └── evaluate_model.py       # Accuracy, Precision, Recall, F1, Confusion Matrix
│   ├── api/
│   │   └── app.py                  # Flask REST API with CORS
│   ├── saved_models/               # Trained model artifacts (.pkl)
│   ├── main.py                     # Pipeline orchestrator
│   └── requirements.txt
├── frontend/                       # Web Dashboard
│   ├── index.html                  # Premium SPA dashboard
│   ├── css/styles.css              # Dark-mode glassmorphism design
│   └── js/app.js                   # API integration & dynamic UI
├── customer_support_tickets.csv    # Real-world dataset (~8,400 tickets)
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the ML Models

```bash
cd backend
python main.py
```

This will:
- Load and preprocess the customer support ticket dataset
- Extract TF-IDF + structured features
- Train category classifier (LinearSVC) and priority classifier (RandomForest)
- Evaluate models and save metrics
- Save trained models to `backend/saved_models/`

### 3. Start the API Server

```bash
cd backend
python -m api.app
```

API runs at `http://localhost:5000`:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify a ticket → category + priority + confidence |
| `/metrics` | GET | Model evaluation metrics |
| `/health` | GET | API health check |

### 4. Open the Frontend

Open `frontend/index.html` in your browser. The dashboard connects to the API on `localhost:5000`.

---

## 🤖 ML Pipeline

| Step | Module | Description |
|------|--------|-------------|
| **1. Preprocessing** | `text_preprocessor.py` | Lowercasing, HTML removal, stopword removal, lemmatization |
| **2. Feature Extraction** | `feature_extractor.py` | TF-IDF (5000 features, bigrams) + structured metadata features |
| **3. Category Model** | `train_model.py` | LinearSVC with calibrated probabilities, balanced class weights |
| **4. Priority Model** | `train_model.py` | RandomForest (200 trees), balanced class weights |
| **5. Evaluation** | `evaluate_model.py` | Per-class precision, recall, F1, confusion matrix |

### Categories (5 classes)
`Billing inquiry` · `Technical issue` · `Cancellation request` · `Product inquiry` · `Refund request`

### Priority Levels (4 classes)
`Critical` · `High` · `Medium` · `Low`

---

## 🎨 Frontend Features

- **Dark-mode glassmorphism** design with gradient accents
- **One-click ticket classification** with animated confidence bars
- **Per-class probability breakdown** for both category and priority
- **Model performance dashboard** with accuracy rings, metrics tables, confusion matrices
- **Quick example chips** to test different ticket types instantly
- **Fully responsive** design for desktop and mobile

---

## 📊 API Usage Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject": "Payment issue", "description": "I was charged twice for my subscription"}'
```

Response:
```json
{
  "success": true,
  "prediction": {
    "category": "Billing inquiry",
    "category_confidence": 0.45,
    "priority": "High",
    "priority_confidence": 0.38
  }
}
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn |
| NLP | NLTK, TF-IDF |
| API | Flask + Flask-CORS |
| Data | Pandas, NumPy |
| Frontend | HTML5, CSS3, Vanilla JS |
| Serialization | Joblib |

---

## 📝 How This Improves Support Operations

1. **Faster Routing** — Tickets automatically go to the right department
2. **Priority Triage** — Critical issues surface immediately
3. **Reduced Manual Sorting** — Support agents focus on solving, not sorting
4. **Scalable** — Handles thousands of tickets without human bottleneck
5. **Data-Driven** — Confidence scores help managers audit and improve
