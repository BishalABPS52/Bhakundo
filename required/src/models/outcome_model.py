"""Match Outcome Prediction Model - Predicts Home Win / Draw / Away Win"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report

from src.config import MODEL_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class OutcomeModel:
    """
    Ensemble model for predicting match outcomes (H/D/A)
    Uses XGBoost, LightGBM, and CatBoost with weighted voting
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize base models with configurations"""
        logger.info("Initializing outcome prediction models...")
        
        # XGBoost
        xgb_config = MODEL_CONFIG['outcome_model']['xgboost']
        self.models['xgb'] = XGBClassifier(**xgb_config)
        
        # LightGBM
        lgbm_config = MODEL_CONFIG['outcome_model']['lightgbm']
        self.models['lgbm'] = LGBMClassifier(**lgbm_config)
        
        # CatBoost
        catboost_config = MODEL_CONFIG['outcome_model']['catboost']
        self.models['catboost'] = CatBoostClassifier(**catboost_config)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train all models and create ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels ('H', 'D', 'A')
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training outcome models with {len(X_train)} samples...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
        
        metrics = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                if name == 'xgb' and X_val is not None:
                    model.fit(
                        X_train, y_train_encoded,
                        eval_set=[(X_val, y_val_encoded)],
                        verbose=False
                    )
                elif name == 'lgbm' and X_val is not None:
                    model.fit(
                        X_train, y_train_encoded,
                        eval_set=[(X_val, y_val_encoded)],
                        callbacks=[
                            # LightGBM callback for early stopping
                        ]
                    )
                elif name == 'catboost' and X_val is not None:
                    model.fit(
                        X_train, y_train_encoded,
                        eval_set=(X_val, y_val_encoded),
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train_encoded)
                
                # Evaluate on validation set
                if X_val is not None:
                    val_pred = model.predict(X_val)
                    val_acc = accuracy_score(y_val_encoded, val_pred)
                    
                    val_pred_proba = model.predict_proba(X_val)
                    val_logloss = log_loss(y_val_encoded, val_pred_proba)
                    
                    metrics[f'{name}_val_accuracy'] = val_acc
                    metrics[f'{name}_val_logloss'] = val_logloss
                    
                    logger.info(f"{name} - Val Accuracy: {val_acc:.4f}, Val LogLoss: {val_logloss:.4f}")
            
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Create weighted ensemble
        self._create_ensemble()
        
        self.is_trained = True
        logger.info("All models trained successfully")
        
        return metrics
    
    def _create_ensemble(self):
        """Create weighted voting ensemble"""
        weights = MODEL_CONFIG['ensemble_weights']
        
        estimators = [
            ('xgb', self.models['xgb']),
            ('lgbm', self.models['lgbm']),
            ('catboost', self.models['catboost'])
        ]
        
        # Get weights in same order as estimators
        weight_list = [
            weights['xgboost'],
            weights['lightgbm'],
            weights['catboost']
        ]
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use predicted probabilities
            weights=weight_list
        )
        
        # Note: ensemble.fit() is not needed since individual models are already fitted
        # We'll use manual prediction aggregation instead
        
        logger.info(f"Ensemble created with weights: {weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict match outcomes
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted outcomes (encoded)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions from all models
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            
            # Get weight for this model
            if name == 'xgb':
                weight = MODEL_CONFIG['ensemble_weights']['xgboost']
            elif name == 'lgbm':
                weight = MODEL_CONFIG['ensemble_weights']['lightgbm']
            else:
                weight = MODEL_CONFIG['ensemble_weights']['catboost']
            
            weights.append(weight)
        
        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Simple weighted majority vote
        final_predictions = []
        for i in range(len(X)):
            votes = predictions[:, i]
            weighted_votes = {}
            
            for vote, weight in zip(votes, weights):
                weighted_votes[vote] = weighted_votes.get(vote, 0) + weight
            
            final_pred = max(weighted_votes, key=weighted_votes.get)
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each outcome
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 3) with probabilities for H/D/A
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get probability predictions from all models
        probas = []
        weights = []
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            probas.append(proba)
            
            # Get weight
            if name == 'xgb':
                weight = MODEL_CONFIG['ensemble_weights']['xgboost']
            elif name == 'lgbm':
                weight = MODEL_CONFIG['ensemble_weights']['lightgbm']
            else:
                weight = MODEL_CONFIG['ensemble_weights']['catboost']
            
            weights.append(weight)
        
        # Weighted average of probabilities
        probas = np.array(probas)
        weights = np.array(weights).reshape(-1, 1, 1)
        
        weighted_proba = np.sum(probas * weights, axis=0) / np.sum(weights)
        
        return weighted_proba
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from XGBoost model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Use XGBoost feature importance
        importance = self.models['xgb'].feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance
    
    def save_model(self, filepath: Optional[Path] = None):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = MODELS_DIR / "outcome_model.pkl"
        
        model_data = {
            'models': self.models,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Optional[Path] = None):
        """Load trained model from disk"""
        if filepath is None:
            filepath = MODELS_DIR / "outcome_model.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        self._create_ensemble()
        
        logger.info(f"Model loaded from {filepath}")
    
    def decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert encoded predictions back to labels"""
        return self.label_encoder.inverse_transform(predictions)


if __name__ == "__main__":
    # Test the outcome model
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice(['H', 'D', 'A'], n_samples))
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    model = OutcomeModel()
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    print("\nTraining metrics:", metrics)
    
    # Test prediction
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"\nSample probabilities:\n{probabilities[:5]}")
