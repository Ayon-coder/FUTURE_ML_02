"""
Main Pipeline Orchestrator
==========================
Runs the complete ML pipeline:
1. Load real customer support ticket dataset
2. Preprocess text data
3. Extract TF-IDF + structured features
4. Train category + priority classifiers
5. Evaluate models
6. Save trained models and metrics
"""

import os
import sys
import pandas as pd

# Add backend root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.text_preprocessor import preprocess_dataframe
from features.feature_extractor import FeatureExtractor
from models.train_model import TicketClassifierTrainer
from models.evaluate_model import evaluate_all


def run_pipeline():
    """Execute the full ML training pipeline."""
    
    print("\n" + "=" * 70)
    print("  🎫  SUPPORT TICKET CLASSIFIER — TRAINING PIPELINE")
    print("=" * 70)
    
    # ──────────────────────────────────────────────────────────
    # STEP 1: Load Dataset
    # ──────────────────────────────────────────────────────────
    print("\n📂 STEP 1: Loading dataset...")
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'customer_support_tickets.csv')
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        print(f"  ✗ Dataset not found at: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"  → Loaded {len(df)} tickets")
    print(f"  → Columns: {list(df.columns)}")
    
    # ──────────────────────────────────────────────────────────
    # STEP 2: Filter & Clean Data
    # ──────────────────────────────────────────────────────────
    print("\n🧹 STEP 2: Filtering and cleaning data...")
    
    required_cols = ['Ticket Type', 'Ticket Priority', 'Ticket Subject', 'Ticket Description']
    for col in required_cols:
        if col not in df.columns:
            print(f"  ✗ Missing required column: {col}")
            sys.exit(1)
    
    # Drop rows with missing category or priority
    initial_count = len(df)
    df = df.dropna(subset=['Ticket Type', 'Ticket Priority']).reset_index(drop=True)
    print(f"  → Dropped {initial_count - len(df)} rows with missing labels")
    
    # Show class distribution
    print(f"\n  Category Distribution:")
    for cat, count in df['Ticket Type'].value_counts().items():
        print(f"    {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  Priority Distribution:")
    for pri, count in df['Ticket Priority'].value_counts().items():
        print(f"    {pri}: {count} ({count/len(df)*100:.1f}%)")
    
    # ──────────────────────────────────────────────────────────
    # STEP 3: Preprocess Text
    # ──────────────────────────────────────────────────────────
    print("\n📝 STEP 3: Preprocessing text data...")
    df = preprocess_dataframe(df)
    
    # ──────────────────────────────────────────────────────────
    # STEP 4: Extract Features (TF-IDF + Structured)
    # ──────────────────────────────────────────────────────────
    print("\n🔢 STEP 4: Extracting TF-IDF + structured features...")
    
    feature_extractor = FeatureExtractor(
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )
    
    X = feature_extractor.fit_transform(df['processed_text'], df=df)
    y_category = df['Ticket Type'].values
    y_priority = df['Ticket Priority'].values
    
    # ──────────────────────────────────────────────────────────
    # STEP 5: Train Models
    # ──────────────────────────────────────────────────────────
    print("\n🤖 STEP 5: Training classification models...")
    
    trainer = TicketClassifierTrainer(random_state=42)
    trainer.prepare_data(X, y_category, y_priority, test_size=0.2)
    training_results = trainer.train_all()
    
    # ──────────────────────────────────────────────────────────
    # STEP 6: Evaluate Models
    # ──────────────────────────────────────────────────────────
    print("\n📊 STEP 6: Evaluating models...")
    
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    metrics = evaluate_all(trainer, save_dir=save_dir)
    
    # ──────────────────────────────────────────────────────────
    # STEP 7: Save Models
    # ──────────────────────────────────────────────────────────
    print("\n💾 STEP 7: Saving trained models...")
    
    trainer.save_models(save_dir)
    feature_extractor.save(os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
    
    # ──────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  Category Accuracy:  {metrics['category']['accuracy']:.4f}")
    print(f"  Priority Accuracy:  {metrics['priority']['accuracy']:.4f}")
    print(f"  Models saved to:    {save_dir}/")
    print(f"\n  Files saved:")
    for f in os.listdir(save_dir):
        fpath = os.path.join(save_dir, f)
        size = os.path.getsize(fpath)
        print(f"    • {f} ({size/1024:.1f} KB)")
    print()
    
    return metrics


if __name__ == "__main__":
    run_pipeline()
