"""
Feature Extraction Module
=========================
Converts preprocessed text into numerical features using TF-IDF vectorization.
Also extracts structured features from ticket metadata columns.
Supports fitting, transforming, and persisting the vectorizer.
"""

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class FeatureExtractor:
    """
    Combined feature extractor for support tickets.
    Extracts:
      1. TF-IDF features from processed text
      2. TF-IDF features from ticket subject
      3. Engineered features: product, channel, customer age, ticket subject category
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), max_df=0.95, min_df=2):
        # Text vectorizer
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w\w+\b'
        )
        
        # Subject-specific vectorizer (captures ticket topic keywords)
        self.subject_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w\w+\b'
        )
        
        # Encoders for categorical/structured features
        self.product_encoder = LabelEncoder()
        self.channel_encoder = LabelEncoder()
        self.subject_cat_encoder = LabelEncoder()
        self.age_scaler = StandardScaler()
        
        self.is_fitted = False
    
    def fit_transform(self, texts, df=None):
        """
        Fit and transform text + structured features.
        
        Parameters
        ----------
        texts : array-like of str
            Preprocessed text data.
        df : pandas.DataFrame, optional
            Original DataFrame with structured columns.
        
        Returns
        -------
        sparse matrix
            Combined feature matrix.
        """
        # TF-IDF on full text
        text_features = self.text_vectorizer.fit_transform(texts)
        
        if df is not None:
            # TF-IDF on subject
            subjects = df['Ticket Subject'].fillna('').values
            subject_features = self.subject_vectorizer.fit_transform(subjects)
            
            # Structured features
            struct_features = self._extract_structured_features(df, fit=True)
            
            combined = hstack([text_features, subject_features, struct_features])
        else:
            combined = text_features
        
        self.is_fitted = True
        print(f"  → Feature matrix shape: {combined.shape}")
        return combined
    
    def transform(self, texts, df=None):
        """
        Transform new data using fitted vectorizers.
        
        Parameters
        ----------
        texts : array-like of str
            Preprocessed text data.
        df : pandas.DataFrame, optional
            DataFrame with structured columns.
        
        Returns
        -------
        sparse matrix
            Combined feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first.")
        
        text_features = self.text_vectorizer.transform(texts)
        
        if df is not None:
            subjects = df['Ticket Subject'].fillna('').values
            subject_features = self.subject_vectorizer.transform(subjects)
            struct_features = self._extract_structured_features(df, fit=False)
            combined = hstack([text_features, subject_features, struct_features])
        else:
            combined = text_features
        
        return combined
    
    def transform_single(self, processed_text, subject="", product="unknown", channel="unknown", age=30):
        """
        Transform a single ticket for API prediction.
        
        Parameters
        ----------
        processed_text : str
            Preprocessed text.
        subject : str
            Ticket subject.
        product : str
            Product purchased.
        channel : str
            Ticket channel.
        age : int
            Customer age.
        
        Returns
        -------
        sparse matrix
            Feature vector for one ticket.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first.")
        
        text_features = self.text_vectorizer.transform([processed_text])
        subject_features = self.subject_vectorizer.transform([subject])
        
        # Encode structured features safely
        product_enc = self._safe_encode(self.product_encoder, product)
        channel_enc = self._safe_encode(self.channel_encoder, channel)
        subject_cat_enc = self._safe_encode(self.subject_cat_encoder, subject)
        age_scaled = self.age_scaler.transform([[age]])[0][0]
        
        struct = np.array([[product_enc, channel_enc, subject_cat_enc, age_scaled]])
        struct_sparse = csr_matrix(struct)
        
        return hstack([text_features, subject_features, struct_sparse])
    
    def _extract_structured_features(self, df, fit=False):
        """Extract and encode structured features from DataFrame."""
        import pandas as pd
        
        features = []
        
        # Product
        products = df['Product Purchased'].fillna('unknown').values if 'Product Purchased' in df.columns else ['unknown'] * len(df)
        if fit:
            product_enc = self.product_encoder.fit_transform(products)
        else:
            product_enc = self._safe_encode_array(self.product_encoder, products)
        features.append(product_enc.reshape(-1, 1))
        
        # Channel
        channels = df['Ticket Channel'].fillna('unknown').values if 'Ticket Channel' in df.columns else ['unknown'] * len(df)
        if fit:
            channel_enc = self.channel_encoder.fit_transform(channels)
        else:
            channel_enc = self._safe_encode_array(self.channel_encoder, channels)
        features.append(channel_enc.reshape(-1, 1))
        
        # Subject category 
        subjects = df['Ticket Subject'].fillna('unknown').values if 'Ticket Subject' in df.columns else ['unknown'] * len(df)
        if fit:
            subject_enc = self.subject_cat_encoder.fit_transform(subjects)
        else:
            subject_enc = self._safe_encode_array(self.subject_cat_encoder, subjects)
        features.append(subject_enc.reshape(-1, 1))
        
        # Age (scaled)
        ages = df['Customer Age'].fillna(30).values.reshape(-1, 1) if 'Customer Age' in df.columns else np.full((len(df), 1), 30)
        if fit:
            age_scaled = self.age_scaler.fit_transform(ages)
        else:
            age_scaled = self.age_scaler.transform(ages)
        features.append(age_scaled)
        
        struct_array = np.hstack(features)
        return csr_matrix(struct_array)
    
    def _safe_encode(self, encoder, value):
        """Safely encode a single value, returning 0 for unseen labels."""
        try:
            return int(encoder.transform([value])[0])
        except (ValueError, KeyError):
            return 0
    
    def _safe_encode_array(self, encoder, values):
        """Safely encode an array of values."""
        result = np.zeros(len(values), dtype=int)
        for i, v in enumerate(values):
            result[i] = self._safe_encode(encoder, v)
        return result
    
    def get_feature_names(self):
        """Return the text feature names."""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first.")
        return self.text_vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save all extractors to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'text_vectorizer': self.text_vectorizer,
            'subject_vectorizer': self.subject_vectorizer,
            'product_encoder': self.product_encoder,
            'channel_encoder': self.channel_encoder,
            'subject_cat_encoder': self.subject_cat_encoder,
            'age_scaler': self.age_scaler,
        }
        joblib.dump(data, filepath)
        print(f"  → Feature extractor saved to {filepath}")
    
    def load(self, filepath):
        """Load all extractors from disk."""
        data = joblib.load(filepath)
        self.text_vectorizer = data['text_vectorizer']
        self.subject_vectorizer = data['subject_vectorizer']
        self.product_encoder = data['product_encoder']
        self.channel_encoder = data['channel_encoder']
        self.subject_cat_encoder = data['subject_cat_encoder']
        self.age_scaler = data['age_scaler']
        self.is_fitted = True
        print(f"  → Feature extractor loaded from {filepath}")
        return self
