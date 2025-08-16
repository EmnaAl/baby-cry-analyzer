import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class BabyCryClassifier:
    """
    Machine Learning model for classifying baby cries into different need categories.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of ML model ('random_forest', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.classes = ['hungry', 'need_to_change', 'pain', 'tired']
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_data(self, features: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            features: List of feature vectors
            labels: List of corresponding labels
            
        Returns:
            Tuple of (X, y) prepared for training
        """
        X = np.array(features)
        y = np.array(labels)
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results dictionary
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Fit label encoder
        self.label_encoder.fit(y)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train_encoded)
        val_score = self.model.score(X_val_scaled, y_val_encoded)
        
        # Predictions for detailed evaluation
        y_val_pred = self.model.predict(X_val_scaled)
        y_val_pred_labels = self.label_encoder.inverse_transform(y_val_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=5)
        
        results = {
            'train_accuracy': train_score,
            'validation_accuracy': val_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_val, y_val_pred_labels),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred_labels),
            'feature_importance': self._get_feature_importance()
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Convert back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get confidence scores (max probability for each prediction)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions.tolist(), confidence_scores.tolist()
    
    def predict_single(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict for a single audio sample with detailed results.
        
        Args:
            features: Feature vector for single audio sample
            
        Returns:
            Dictionary with prediction details
        """
        predictions, confidence = self.predict(features)
        
        # Get probabilities for all classes
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create probability dictionary
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(probabilities[i])
        
        # Sort by probability
        sorted_predictions = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'predicted_need': predictions[0],
            'confidence': float(confidence[0]),
            'all_probabilities': class_probabilities,
            'top_predictions': sorted_predictions[:3]
        }
    
    def _get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'classes': self.classes
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.classes = model_data.get('classes', self.classes)
        
        print(f"Model loaded from {model_path}")
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Best parameters and scores
        """
        # Prepare data
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_scaled, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        return {
            'model_type': self.model_type,
            'classes': self.classes,
            'num_classes': len(self.classes),
            'is_trained': self.model is not None,
            'model_params': self.model.get_params() if self.model else None
        }
