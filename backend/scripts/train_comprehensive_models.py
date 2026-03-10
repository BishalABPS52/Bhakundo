"""
COMPREHENSIVE MODEL TRAINING PIPELINE
Trains 3-model ensemble system for Premier League predictions:
1. Base Model (CatBoost Classifier) - Outcome probabilities
2. Score Model (2x CatBoost Regressors) - Home/Away goals → Poisson probabilities  
3. Lineup Model (CatBoost Classifier) - When lineups available

Uses 1051+ matches from 2019-2025 with 125+ features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comprehensive_feature_engineering import ComprehensiveFeatureEngineer

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'required' / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'pl'
MODELS_DIR = DATA_DIR / 'models'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_all_historical_data():
    """
    Load all historical Premier League data from 2019-2025
    Target: ~1051 completed matches
    """
    logger.info("="*100)
    logger.info("LOADING HISTORICAL DATA (2019-2025)")
    logger.info("="*100)
    
    data_files = [
        RAW_DIR / 'matches_2019_2022.csv',
        RAW_DIR / 'matches_2023.csv',
        RAW_DIR / 'matches_2024.csv',
        RAW_DIR / 'matches_2025.csv',
        RAW_DIR / 'pl_2023_historical.csv',
        RAW_DIR / 'pl_2024_historical.csv',
        RAW_DIR / 'pl_2025_26_completed_matches.csv'
    ]
    
    dfs = []
    
    for file in data_files:
        if file.exists():
            df = pd.read_csv(file)
            
            # Standardize column names
            if 'api_match_id' in df.columns and 'match_id' not in df.columns:
                df = df.rename(columns={'api_match_id': 'match_id'})
            
            # Standardize score columns
            if 'home_score' in df.columns and 'home_goals' not in df.columns:
                df['home_goals'] = df['home_score']
                df['away_goals'] = df['away_score']
            
            # Filter only completed matches
            if 'status' in df.columns:
                df = df[df['status'].isin(['FINISHED', 'FT'])].copy()
            elif 'result' in df.columns:
                df = df[df['result'].isin(['H', 'D', 'A'])].copy()
            
            # Ensure required columns exist
            required_cols = ['match_id', 'home_team', 'away_team', 'home_goals', 'away_goals']
            if all(col in df.columns for col in required_cols):
                # Parse match_date
                if 'match_date' in df.columns:
                    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
                else:
                    df['match_date'] = pd.Timestamp('2024-01-01')
                
                df = df.dropna(subset=['home_goals', 'away_goals'])
                df['home_goals'] = df['home_goals'].astype(int)
                df['away_goals'] = df['away_goals'].astype(int)
                
                logger.info(f"  ✅ Loaded {len(df):4d} matches from {file.name}")
                dfs.append(df[required_cols + ['match_date']])
            else:
                logger.warning(f"  ⚠️  Skipping {file.name} - missing required columns")
        else:
            logger.warning(f"  ❌ File not found: {file.name}")
    
    if not dfs:
        raise ValueError("No data files found! Cannot train models.")
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['match_id'], keep='last')
    duplicates_removed = initial_count - len(combined_df)
    if duplicates_removed > 0:
        logger.info(f"  🗑️  Removed {duplicates_removed} duplicate matches")
    
    # Sort by date
    combined_df = combined_df.sort_values('match_date').reset_index(drop=True)
    
    logger.info(f"✅ TOTAL UNIQUE COMPLETED MATCHES: {len(combined_df)}")
    logger.info(f"   Date range: {combined_df['match_date'].min()} to {combined_df['match_date'].max()}")
    
    return combined_df


def train_base_outcome_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    Train CatBoost classifier for match outcome (Home/Draw/Away)
    """
    logger.info("\n" + "="*100)
    logger.info("TRAINING BASE OUTCOME MODEL (CatBoost Classifier)")
    logger.info("="*100)
    
    # Class weights for balanced training
    class_counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    logger.info(f"Class distribution: {dict(zip(['Away Win', 'Draw', 'Home Win'], class_counts))}")
    logger.info(f"Class weights: {class_weights}")
    
    # CatBoost model with optimized hyperparameters
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        class_weights=list(class_weights.values())
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\n✅ Base Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
    
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 20 Most Important Features:")
    logger.info(feature_importance.head(20).to_string(index=False))
    
    return model, accuracy


def train_score_models(X_train, home_goals_train, away_goals_train, X_test, home_goals_test, away_goals_test, feature_cols):
    """
    Train 2 regression models for home and away goals
    These will be converted to probabilities using Poisson distribution
    """
    logger.info("\n" + "="*100)
    logger.info("TRAINING SCORE MODELS (CatBoost Regressors)")
    logger.info("="*100)
    
    # Home goals model
    logger.info("\n📊 Training HOME GOALS model...")
    home_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3,
        loss_function='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    home_model.fit(
        X_train, home_goals_train,
        eval_set=(X_test, home_goals_test),
        use_best_model=True
    )
    
    home_pred = home_model.predict(X_test)
    home_mae = mean_absolute_error(home_goals_test, home_pred)
    home_rmse = np.sqrt(mean_squared_error(home_goals_test, home_pred))
    
    logger.info(f"✅ Home Goals Model - MAE: {home_mae:.4f}, RMSE: {home_rmse:.4f}")
    
    # Away goals model
    logger.info("\n📊 Training AWAY GOALS model...")
    away_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3,
        loss_function='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    away_model.fit(
        X_train, away_goals_train,
        eval_set=(X_test, away_goals_test),
        use_best_model=True
    )
    
    away_pred = away_model.predict(X_test)
    away_mae = mean_absolute_error(away_goals_test, away_pred)
    away_rmse = np.sqrt(mean_squared_error(away_goals_test, away_pred))
    
    logger.info(f"✅ Away Goals Model - MAE: {away_mae:.4f}, RMSE: {away_rmse:.4f}")
    
    # Test Poisson conversion
    logger.info("\n🎲 Testing Poisson probability conversion...")
    test_sample = 10
    for i in range(min(test_sample, len(X_test))):
        h_xg = home_pred[i]
        a_xg = away_pred[i]
        
        # Calculate outcome probabilities using Poisson
        outcome_probs = calculate_poisson_outcome_probs(h_xg, a_xg)
        
        logger.info(f"  Match {i+1}: xG {h_xg:.2f}-{a_xg:.2f} → H:{outcome_probs['home']:.2%} D:{outcome_probs['draw']:.2%} A:{outcome_probs['away']:.2%}")
    
    return home_model, away_model, home_mae, away_mae


def calculate_poisson_outcome_probs(home_xg, away_xg, max_goals=7):
    """
    Convert expected goals to outcome probabilities using Poisson distribution
    """
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    
    # Apply home advantage adjustment
    home_xg_adj = home_xg * 1.10
    away_xg_adj = away_xg * 0.95
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_home = poisson.pmf(home_goals, home_xg_adj)
            prob_away = poisson.pmf(away_goals, away_xg_adj)
            combined_prob = prob_home * prob_away
            
            if home_goals > away_goals:
                home_win_prob += combined_prob
            elif home_goals < away_goals:
                away_win_prob += combined_prob
            else:
                draw_prob += combined_prob
    
    return {
        'home': home_win_prob,
        'draw': draw_prob,
        'away': away_win_prob
    }


def train_lineup_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    Train lineup-based model (when lineups are available)
    For now, this is similar to base model but can be enhanced with lineup features
    """
    logger.info("\n" + "="*100)
    logger.info("TRAINING LINEUP MODEL (CatBoost Classifier)")
    logger.info("="*100)
    logger.info("Note: Using same features as base model. Can be enhanced with lineup data.")
    
    # Class weights
    class_counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=43,
        verbose=100,
        early_stopping_rounds=50,
        class_weights=list(class_weights.values())
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\n✅ Lineup Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, accuracy


def save_models(base_model, home_model, away_model, lineup_model, feature_cols):
    """
    Save all trained models
    """
    logger.info("\n" + "="*100)
    logger.info("SAVING MODELS")
    logger.info("="*100)
    
    # Save base model
    base_path = MODELS_DIR / 'pl_base_outcome_model.pkl'
    joblib.dump({
        'model': base_model,
        'feature_columns': feature_cols,
        'model_type': 'catboost_classifier',
        'trained_date': datetime.now().isoformat()
    }, base_path)
    logger.info(f"✅ Base model saved: {base_path}")
    
    # Save score models
    score_path = MODELS_DIR / 'pl_score_prediction_model.pkl'
    joblib.dump({
        'home_model': home_model,
        'away_model': away_model,
        'feature_columns': feature_cols,
        'model_type': 'catboost_regressor_pair',
        'trained_date': datetime.now().isoformat()
    }, score_path)
    logger.info(f"✅ Score models saved: {score_path}")
    
    # Save lineup model
    lineup_path = MODELS_DIR / 'pl_lineup_model.pkl'
    joblib.dump({
        'model': lineup_model,
        'feature_columns': feature_cols,
        'model_type': 'catboost_classifier',
        'trained_date': datetime.now().isoformat()
    }, lineup_path)
    logger.info(f"✅ Lineup model saved: {lineup_path}")
    
    logger.info(f"\n🎉 ALL MODELS SAVED TO: {MODELS_DIR}")


def main():
    """
    Main training pipeline
    """
    start_time = datetime.now()
    
    logger.info("\n" + "="*100)
    logger.info("🚀 COMPREHENSIVE PREMIER LEAGUE PREDICTION MODEL TRAINING")
    logger.info("="*100)
    logger.info(f"Start time: {start_time}")
    
    # 1. Load historical data
    matches_df = load_all_historical_data()
    
    # 2. Build features
    engineer = ComprehensiveFeatureEngineer()
    training_df, feature_cols = engineer.build_training_dataset(matches_df)
    
    logger.info(f"\n📊 Training dataset shape: {training_df.shape}")
    logger.info(f"📊 Number of features: {len(feature_cols)}")
    
    # 3. Prepare training data
    X = training_df[feature_cols]
    y_outcome = training_df['outcome']
    y_home_goals = training_df['home_goals']
    y_away_goals = training_df['away_goals']
    
    # Split data (80/20)
    X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(
        X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome
    )
    
    _, _, y_home_train, y_home_test = train_test_split(
        X, y_home_goals, test_size=0.2, random_state=42
    )
    
    _, _, y_away_train, y_away_test = train_test_split(
        X, y_away_goals, test_size=0.2, random_state=42
    )
    
    logger.info(f"\n📊 Training set: {len(X_train)} matches")
    logger.info(f"📊 Test set: {len(X_test)} matches")
    
    # 4. Train models
    base_model, base_acc = train_base_outcome_model(
        X_train, y_outcome_train, X_test, y_outcome_test, feature_cols
    )
    
    home_model, away_model, home_mae, away_mae = train_score_models(
        X_train, y_home_train, y_away_train, X_test, y_home_test, y_away_test, feature_cols
    )
    
    lineup_model, lineup_acc = train_lineup_model(
        X_train, y_outcome_train, X_test, y_outcome_test, feature_cols
    )
    
    # 5. Save models
    save_models(base_model, home_model, away_model, lineup_model, feature_cols)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*100)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("="*100)
    logger.info(f"Total matches trained: {len(training_df)}")
    logger.info(f"Features used: {len(feature_cols)}")
    logger.info(f"Base model accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")
    logger.info(f"Score models MAE: Home={home_mae:.4f}, Away={away_mae:.4f}")
    logger.info(f"Lineup model accuracy: {lineup_acc:.4f} ({lineup_acc*100:.2f}%)")
    logger.info(f"Training duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("="*100)


if __name__ == "__main__":
    main()
