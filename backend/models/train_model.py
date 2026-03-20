"""
Model Training Module
=====================
Trains two classification models:
1. Category Classifier — LinearSVC for predicting ticket type
   (Technical issue, Billing inquiry, Cancellation request, Product inquiry, Refund request)
2. Priority Classifier — RandomForestClassifier for predicting priority level
   (Critical, High, Medium, Low)

Uses 80/20 train-test split and saves trained models as .pkl files.
"""

import os
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TicketClassifierTrainer:
    """
    Trains and manages category and priority classification models.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Category classifier: LinearSVC with calibration for probability estimates
        self.category_model = CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                max_iter=10000,
                random_state=random_state,
                class_weight='balanced'
            ),
            cv=3
        )
        
        # Priority classifier: RandomForest
        self.priority_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Label encoders
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        
        # Store split data for evaluation
        self.X_train = None
        self.X_test = None
        self.y_cat_train = None
        self.y_cat_test = None
        self.y_pri_train = None
        self.y_pri_test = None
    
    def prepare_data(self, X, y_category, y_priority, test_size=0.2):
        """
        Split data into train/test sets and encode labels.
        
        Parameters
        ----------
        X : sparse matrix
            TF-IDF feature matrix.
        y_category : array-like
            Category labels.
        y_priority : array-like
            Priority labels.
        test_size : float
            Fraction of data to use for testing.
        
        Returns
        -------
        dict
            Split data information.
        """
        # Encode labels
        y_cat_encoded = self.category_encoder.fit_transform(y_category)
        y_pri_encoded = self.priority_encoder.fit_transform(y_priority)
        
        # Split data
        self.X_train, self.X_test, self.y_cat_train, self.y_cat_test, self.y_pri_train, self.y_pri_test = \
            train_test_split(
                X, y_cat_encoded, y_pri_encoded,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y_cat_encoded
            )
        
        split_info = {
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0],
            'category_classes': list(self.category_encoder.classes_),
            'priority_classes': list(self.priority_encoder.classes_),
            'feature_count': self.X_train.shape[1]
        }
        
        print(f"  → Train size: {split_info['train_size']}, Test size: {split_info['test_size']}")
        print(f"  → Categories: {split_info['category_classes']}")
        print(f"  → Priorities: {split_info['priority_classes']}")
        
        return split_info
    
    def train_category_model(self):
        """Train the category classification model."""
        if self.X_train is None:
            raise RuntimeError("Call prepare_data() first.")
        
        print("\n  → Training Category Classifier (LinearSVC + Calibration)...")
        self.category_model.fit(self.X_train, self.y_cat_train)
        
        train_acc = self.category_model.score(self.X_train, self.y_cat_train)
        test_acc = self.category_model.score(self.X_test, self.y_cat_test)
        
        print(f"    Train Accuracy: {train_acc:.4f}")
        print(f"    Test Accuracy:  {test_acc:.4f}")
        
        return {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    
    def train_priority_model(self):
        """Train the priority classification model."""
        if self.X_train is None:
            raise RuntimeError("Call prepare_data() first.")
        
        print("\n  → Training Priority Classifier (RandomForest)...")
        self.priority_model.fit(self.X_train, self.y_pri_train)
        
        train_acc = self.priority_model.score(self.X_train, self.y_pri_train)
        test_acc = self.priority_model.score(self.X_test, self.y_pri_test)
        
        print(f"    Train Accuracy: {train_acc:.4f}")
        print(f"    Test Accuracy:  {test_acc:.4f}")
        
        return {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    
    def train_all(self):
        """Train both models and return results."""
        cat_results = self.train_category_model()
        pri_results = self.train_priority_model()
        return {
            'category': cat_results,
            'priority': pri_results
        }
    
    def predict(self, X):
        """
        Predict category and priority for new tickets.
        
        Parameters
        ----------
        X : sparse matrix
            TF-IDF features for new tickets.
        
        Returns
        -------
        dict
            Predictions with labels and confidence scores.
        """
        # Category prediction with probabilities
        cat_pred_encoded = self.category_model.predict(X)
        cat_proba = self.category_model.predict_proba(X)
        cat_pred = self.category_encoder.inverse_transform(cat_pred_encoded)
        
        # Priority prediction with probabilities
        pri_pred_encoded = self.priority_model.predict(X)
        pri_proba = self.priority_model.predict_proba(X)
        pri_pred = self.priority_encoder.inverse_transform(pri_pred_encoded)
        
        results = []
        for i in range(len(cat_pred)):
            cat_conf = float(np.max(cat_proba[i]))
            pri_conf = float(np.max(pri_proba[i]))
            
            results.append({
                'category': cat_pred[i],
                'category_confidence': round(cat_conf, 4),
                'category_probabilities': {
                    cls: round(float(prob), 4)
                    for cls, prob in zip(self.category_encoder.classes_, cat_proba[i])
                },
                'priority': pri_pred[i],
                'priority_confidence': round(pri_conf, 4),
                'priority_probabilities': {
                    cls: round(float(prob), 4)
                    for cls, prob in zip(self.priority_encoder.classes_, pri_proba[i])
                }
            })
        
        return results
    
    def save_models(self, save_dir):
        """
        Save all models and encoders to disk.
        
        Parameters
        ----------
        save_dir : str
            Directory to save models.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.category_model, os.path.join(save_dir, 'category_model.pkl'))
        joblib.dump(self.priority_model, os.path.join(save_dir, 'priority_model.pkl'))
        joblib.dump(self.category_encoder, os.path.join(save_dir, 'category_encoder.pkl'))
        joblib.dump(self.priority_encoder, os.path.join(save_dir, 'priority_encoder.pkl'))
        
        print(f"\n  → Models saved to {save_dir}/")
    
    def load_models(self, save_dir):
        """
        Load all models and encoders from disk.
        
        Parameters
        ----------
        save_dir : str
            Directory containing saved models.
        """
        self.category_model = joblib.load(os.path.join(save_dir, 'category_model.pkl'))
        self.priority_model = joblib.load(os.path.join(save_dir, 'priority_model.pkl'))
        self.category_encoder = joblib.load(os.path.join(save_dir, 'category_encoder.pkl'))
        self.priority_encoder = joblib.load(os.path.join(save_dir, 'priority_encoder.pkl'))
        
        print(f"  → Models loaded from {save_dir}/")
        return self
