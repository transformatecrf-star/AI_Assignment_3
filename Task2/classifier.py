"""
Task 2 Deliverable: Ticket Routing Classifier Class
Encapsulates text cleaning, preprocessing, model training, and inference.
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TicketRoutingClassifier:
    def __init__(self, c_val: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.random_state = random_state
        # Preprocessor: TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english'
        )
        # ML Model: Logistic Regression with balanced class weights for fairness
        self.model = LogisticRegression(
            C=c_val,
            max_iter=max_iter,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.classes_ = None

    @staticmethod
    def clean_text(text: str) -> str:
        """Applies robust text cleaning to raw ticket body."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters/punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def fit(self, raw_texts: pd.Series, labels: pd.Series):
        """Fits the preprocessor and ML model strictly on training data."""
        cleaned_texts = raw_texts.apply(self.clean_text)
        X_vec = self.vectorizer.fit_transform(cleaned_texts)
        self.model.fit(X_vec, labels)
        self.classes_ = self.model.classes_
        return self

    def predict(self, raw_texts: pd.Series) -> pd.DataFrame:
        """
        Predicts department labels and returns class probability scores as JSON strings.
        """
        cleaned_texts = raw_texts.apply(self.clean_text)
        X_vec = self.vectorizer.transform(cleaned_texts)
        
        preds = self.model.predict(X_vec)
        probs = self.model.predict_proba(X_vec)
        
        serialized_scores = []
        for prob_arr in probs:
            score_dict = {cls: float(prob) for cls, prob in zip(self.classes_, prob_arr)}
            serialized_scores.append(json.dumps(score_dict))
            
        return pd.DataFrame({
            'recommended_department': preds,
            'class_scores': serialized_scores
        })