"""
Train Score Prediction Model with Poisson Regression
Predicts home_goals and away_goals separately
Ensures alignment with outcome predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'required' / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'pl'
MODELS_DIR = DATA_DIR / 'models'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_score_models():
    """Train home and away goal prediction models"""
    
    logger.info("="*80)
    logger.info("SCORE PREDICTION MODEL TRAINING")
    logger.info("="*80)
    
    # Load processed training data (same features as outcome model)
    processed_file = DATA_DIR / 'processed' / 'training_dataset_full_history.csv'
    
    if not processed_file.exists():
        logger.error(f"Processed data not found: {processed_file}")
        logger.error("Run train_whole_model.py first to generate features!")
        return
    
    df = pd.read_csv(processed_file)
    logger.info(f"Loaded {len(df)} matches with features")
    
    # Prepare features and targets
    feature_cols = [c for c in df.columns if c not in 
                   ['outcome', 'home_goals', 'away_goals', 'match_date', 
                    'home_team', 'away_team', 'gameweek', 'season']]
    
    X = df[feature_cols].values
    y_home = df['home_goals'].values
    y_away = df['away_goals'].values
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(X)}")
    
    # Use last 50 matches as test set (same as outcome model)
    split_idx = len(X) - 50
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
    y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING HOME GOALS MODEL")
    logger.info("="*80)
    
    # Train home goals model with Poisson objective (ideal for count data)
    home_model = XGBRegressor(
        objective='count:poisson',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    home_model.fit(X_train_scaled, y_home_train, verbose=False)
    
    # Predict and evaluate
    y_home_pred = home_model.predict(X_test_scaled)
    y_home_pred_rounded = np.round(y_home_pred).astype(int).clip(0, 9)  # 0-9 goals
    
    mae_home = mean_absolute_error(y_home_test, y_home_pred_rounded)
    exact_home = (y_home_pred_rounded == y_home_test).sum() / len(y_home_test) * 100
    
    logger.info(f"Home Goals MAE: {mae_home:.2f}")
    logger.info(f"Home Goals Exact: {exact_home:.1f}%")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING AWAY GOALS MODEL")
    logger.info("="*80)
    
    # Train away goals model
    away_model = XGBRegressor(
        objective='count:poisson',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    away_model.fit(X_train_scaled, y_away_train, verbose=False)
    
    # Predict and evaluate
    y_away_pred = away_model.predict(X_test_scaled)
    y_away_pred_rounded = np.round(y_away_pred).astype(int).clip(0, 9)
    
    mae_away = mean_absolute_error(y_away_test, y_away_pred_rounded)
    exact_away = (y_away_pred_rounded == y_away_test).sum() / len(y_away_test) * 100
    
    logger.info(f"Away Goals MAE: {mae_away:.2f}")
    logger.info(f"Away Goals Exact: {exact_away:.1f}%")
    
    # Combined score accuracy
    logger.info("\n" + "="*80)
    logger.info("COMBINED SCORE EVALUATION")
    logger.info("="*80)
    
    exact_score = ((y_home_pred_rounded == y_home_test) & 
                   (y_away_pred_rounded == y_away_test)).sum() / len(y_home_test) * 100
    
    # Within 1 goal accuracy
    within_1_home = (np.abs(y_home_pred_rounded - y_home_test) <= 1).sum() / len(y_home_test) * 100
    within_1_away = (np.abs(y_away_pred_rounded - y_away_test) <= 1).sum() / len(y_away_test) * 100
    within_1_both = within_1_home * within_1_away / 100
    
    logger.info(f"Exact Score Accuracy: {exact_score:.1f}%")
    logger.info(f"Home Goals ±1: {within_1_home:.1f}%")
    logger.info(f"Away Goals ±1: {within_1_away:.1f}%")
    
    # Outcome accuracy from score predictions
    y_outcome_test = []
    y_outcome_pred = []
    
    for h_true, a_true, h_pred, a_pred in zip(y_home_test, y_away_test, 
                                                y_home_pred_rounded, y_away_pred_rounded):
        if h_true > a_true:
            y_outcome_test.append('Home Win')
        elif h_true < a_true:
            y_outcome_test.append('Away Win')
        else:
            y_outcome_test.append('Draw')
        
        if h_pred > a_pred:
            y_outcome_pred.append('Home Win')
        elif h_pred < a_pred:
            y_outcome_pred.append('Away Win')
        else:
            y_outcome_pred.append('Draw')
    
    outcome_acc = (np.array(y_outcome_test) == np.array(y_outcome_pred)).sum() / len(y_outcome_test) * 100
    logger.info(f"Outcome Accuracy from Score: {outcome_acc:.1f}%")
    
    # Save models
    logger.info("\n" + "="*80)
    logger.info("SAVING SCORE MODELS")
    logger.info("="*80)
    
    score_model_data = {
        'home_model': home_model,
        'away_model': away_model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'max_goals': 9
    }
    
    joblib.dump(score_model_data, MODELS_DIR / 'pl_score_prediction_model.pkl')
    logger.info(f"✅ Saved score prediction model to pl_score_prediction_model.pkl")
    
    logger.info("\n" + "="*80)
    logger.info("✅ SCORE MODEL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Exact Score: {exact_score:.1f}%")
    logger.info(f"Outcome Accuracy: {outcome_acc:.1f}%")
    logger.info(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    train_score_models()
