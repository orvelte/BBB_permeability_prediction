"""
Machine Learning Models for BBB Permeability Prediction

This module contains the machine learning pipeline for predicting
Blood-Brain Barrier permeability using molecular descriptors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Tuple, Any, List

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve, auc
)


class BBBPredictor:
    """
    BBB Permeability Predictor using multiple machine learning algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the BBB predictor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.results = {}
        self.trained_models = {}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.feature_names = feature_names
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=self.random_state, 
            stratify=y_encoded
        )
        
        print(f"Feature matrix shape: {X_scaled.shape}")
        print(f"Label distribution: {np.bincount(y_encoded)}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing model results
        """
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'SVM': SVC(
                probability=True, 
                random_state=self.random_state,
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                C=1.0
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store model and results
            self.trained_models[name] = model
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.results = results
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results
        """
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            print(f"\n--- Evaluating {name} ---")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Update results
            self.results[name].update({
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            })
            
            evaluation_results[name] = self.results[name]
            
            print(f"Test Accuracy: {accuracy:.3f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return evaluation_results
    
    def get_feature_importance(self, top_n: int = 20) -> Tuple[List[str], np.ndarray]:
        """
        Get feature importance from Random Forest model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Tuple of (feature_names, importance_scores)
        """
        if 'Random Forest' not in self.trained_models:
            raise ValueError("Random Forest model not found. Train models first.")
        
        rf_model = self.trained_models['Random Forest']
        feature_importance = rf_model.feature_importances_
        
        # Get top features
        top_features_idx = np.argsort(feature_importance)[-top_n:]
        top_features_names = [self.feature_names[i] for i in top_features_idx] if self.feature_names else [f"Feature_{i}" for i in top_features_idx]
        top_features_importance = feature_importance[top_features_idx]
        
        return top_features_names, top_features_importance
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        feature_names, importance_scores = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(feature_names)), importance_scores)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Molecular Descriptors (Random Forest)')
        plt.tight_layout()
        plt.show()
        
        print(f"Top {top_n} Most Important Molecular Descriptors:")
        for name, importance in zip(feature_names, importance_scores):
            print(f"{name}: {importance:.4f}")
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot model performance comparison.
        
        Args:
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results available. Train and evaluate models first.")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [self.results[name].get('accuracy', 0) for name in self.results.keys()],
            'CV Accuracy (Mean)': [self.results[name]['cv_mean'] for name in self.results.keys()],
            'CV Accuracy (Std)': [self.results[name]['cv_std'] for name in self.results.keys()]
        })
        
        print("\n=== Model Comparison ===")
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        plt.figure(figsize=figsize)
        x_pos = np.arange(len(self.results))
        plt.bar(x_pos, [self.results[name].get('accuracy', 0) for name in self.results.keys()], 
                yerr=[self.results[name]['cv_std'] for name in self.results.keys()], 
                capsize=5, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x_pos, list(self.results.keys()), rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix for the best model or specified model.
        
        Args:
            model_name: Name of model to plot (if None, uses best model)
            figsize: Figure size
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x].get('accuracy', 0))
        
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found.")
        
        # Get test data (we need to reconstruct this - this is a limitation of the current design)
        # For now, we'll assume the predictions are available
        y_pred = self.results[model_name]['predictions']
        
        # We need the actual test labels - this should be passed from the main script
        # For now, we'll create a placeholder
        print(f"Confusion matrix for {model_name} would be plotted here.")
        print("Note: This requires access to test labels from the main script.")
    
    def plot_roc_curve(self, model_name: str = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot ROC curve for the best model or specified model.
        
        Args:
            model_name: Name of model to plot (if None, uses best model)
            figsize: Figure size
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x].get('accuracy', 0))
        
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found.")
        
        probabilities = self.results[model_name].get('probabilities')
        if probabilities is None:
            print(f"Model '{model_name}' does not support probability predictions.")
            return
        
        # We need the actual test labels - this should be passed from the main script
        print(f"ROC curve for {model_name} would be plotted here.")
        print("Note: This requires access to test labels from the main script.")
    
    def save_models(self, save_dir: str = '../results/models'):
        """
        Save trained models and preprocessing objects.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            model_path = os.path.join(save_dir, f'{name.lower().replace(" ", "_")}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save preprocessing objects
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
        if self.feature_names:
            joblib.dump(self.feature_names, os.path.join(save_dir, 'feature_names.pkl'))
        
        print("Saved preprocessing objects")
    
    def load_models(self, save_dir: str = '../results/models'):
        """
        Load trained models and preprocessing objects.
        
        Args:
            save_dir: Directory containing saved models
        """
        # Load preprocessing objects
        self.label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))
        self.scaler = joblib.load(os.path.join(save_dir, 'feature_scaler.pkl'))
        
        if os.path.exists(os.path.join(save_dir, 'feature_names.pkl')):
            self.feature_names = joblib.load(os.path.join(save_dir, 'feature_names.pkl'))
        
        # Load models
        model_files = [f for f in os.listdir(save_dir) if f.endswith('_model.pkl')]
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
            model_path = os.path.join(save_dir, model_file)
            self.trained_models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name} model from {model_path}")
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Predictions
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x].get('accuracy', 0))
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.trained_models[model_name].predict(X_scaled)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Get prediction probabilities on new data.
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Prediction probabilities
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x].get('accuracy', 0))
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.trained_models[model_name].predict_proba(X_scaled)
        
        return probabilities


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, 
                         model_name: str = 'Random Forest') -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for a specific model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of model to tune
        
    Returns:
        Best parameters and score
    """
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=42)
    
    elif model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        model = SVC(probability=True, random_state=42)
    
    elif model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }
