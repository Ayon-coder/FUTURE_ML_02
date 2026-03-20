"""
Model Evaluation Module
=======================
Computes comprehensive ML evaluation metrics for both classifiers:
- Accuracy, Precision, Recall, F1-Score (per-class and weighted)
- Confusion Matrix
- Classification Report
Outputs results in JSON-serializable format for the API.
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_classifier(y_true, y_pred, class_names, model_name="Model"):
    """
    Evaluate a single classifier and return comprehensive metrics.
    
    Parameters
    ----------
    y_true : array-like
        True encoded labels.
    y_pred : array-like
        Predicted encoded labels.
    class_names : list of str
        Human-readable class names.
    model_name : str
        Name of the model for display.
    
    Returns
    -------
    dict
        JSON-serializable evaluation metrics.
    """
    # Overall metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # Per-class metrics
    precision_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            'precision': round(float(precision_per[i]), 4),
            'recall': round(float(recall_per[i]), 4),
            'f1_score': round(float(f1_per[i]), 4),
            'support': int(np.sum(np.array(y_true) == i))
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()
    
    # Classification report string
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    metrics = {
        'model_name': model_name,
        'accuracy': round(accuracy, 4),
        'precision_weighted': round(precision_weighted, 4),
        'recall_weighted': round(recall_weighted, 4),
        'f1_weighted': round(f1_weighted, 4),
        'per_class': per_class,
        'confusion_matrix': cm_list,
        'class_names': list(class_names),
        'total_samples': int(len(y_true))
    }
    
    # Print report
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*60}")
    print(report_str)
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_all(trainer, save_dir=None):
    """
    Evaluate both category and priority models.
    
    Parameters
    ----------
    trainer : TicketClassifierTrainer
        Trained trainer instance with test data.
    save_dir : str, optional
        Directory to save metrics JSON.
    
    Returns
    -------
    dict
        Combined evaluation metrics for both models.
    """
    # Category evaluation
    cat_pred = trainer.category_model.predict(trainer.X_test)
    cat_metrics = evaluate_classifier(
        trainer.y_cat_test,
        cat_pred,
        trainer.category_encoder.classes_,
        model_name="Category Classifier (LinearSVC)"
    )
    
    # Priority evaluation
    pri_pred = trainer.priority_model.predict(trainer.X_test)
    pri_metrics = evaluate_classifier(
        trainer.y_pri_test,
        pri_pred,
        trainer.priority_encoder.classes_,
        model_name="Priority Classifier (RandomForest)"
    )
    
    all_metrics = {
        'category': cat_metrics,
        'priority': pri_metrics
    }
    
    # Save metrics to JSON
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  → Metrics saved to {metrics_path}")
    
    return all_metrics
