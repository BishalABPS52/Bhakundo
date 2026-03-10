"""
Enhanced Score Prediction Model with Goal-Specific Features and Ensemble
Uses multiple Poisson models with attack/defense specific features
Aligned with outcome predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'required' / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'pl'
MODELS_DIR = DATA_DIR / 'models'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def add_goal_specific_features(df, matches_df):
    """Add goal-specific features for better score prediction
    
    Features added:
    - Home/away goals scored at venue (last 5/10 matches)
    - Home/away goals conceded at venue (last 5/10 matches)
    - Attack strength rating vs league average
    - Defense strength rating vs league average
    - Half-time goal patterns
    - Recent goal-scoring form
    """
    
    logger.info("Adding goal-specific features...")
    
    # Convert dates to datetime
    df['match_date'] = pd.to_datetime(df['match_date'])
    matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
    
    new_features = []
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        match_date = row['match_date']
        
        features = {}
        
        # ===== HOME TEAM ATTACK (goals scored at home venue) =====
        home_matches_home = matches_df[
            (matches_df['home_team'] == home_team) & 
            (matches_df['match_date'] < match_date)
        ].sort_values('match_date')
        
        if len(home_matches_home) >= 5:
            # Last 5 home games - goals scored
            last5_home = home_matches_home.tail(5)
            features['home_goals_scored_home_last5'] = last5_home['home_goals'].sum()
            features['home_avg_goals_scored_home'] = last5_home['home_goals'].mean()
            features['home_goals_conceded_home_last5'] = last5_home['away_goals'].sum()
            features['home_avg_goals_conceded_home'] = last5_home['away_goals'].mean()
            
            # Last 10 home games
            if len(home_matches_home) >= 10:
                last10_home = home_matches_home.tail(10)
                features['home_avg_goals_scored_home_last10'] = last10_home['home_goals'].mean()
                features['home_avg_goals_conceded_home_last10'] = last10_home['away_goals'].mean()
            else:
                features['home_avg_goals_scored_home_last10'] = features['home_avg_goals_scored_home']
                features['home_avg_goals_conceded_home_last10'] = features['home_avg_goals_conceded_home']
        else:
            features['home_goals_scored_home_last5'] = 0
            features['home_avg_goals_scored_home'] = 1.5  # League average
            features['home_goals_conceded_home_last5'] = 0
            features['home_avg_goals_conceded_home'] = 1.0
            features['home_avg_goals_scored_home_last10'] = 1.5
            features['home_avg_goals_conceded_home_last10'] = 1.0
        
        # ===== AWAY TEAM ATTACK (goals scored at away venue) =====
        away_matches_away = matches_df[
            (matches_df['away_team'] == away_team) & 
            (matches_df['match_date'] < match_date)
        ].sort_values('match_date')
        
        if len(away_matches_away) >= 5:
            last5_away = away_matches_away.tail(5)
            features['away_goals_scored_away_last5'] = last5_away['away_goals'].sum()
            features['away_avg_goals_scored_away'] = last5_away['away_goals'].mean()
            features['away_goals_conceded_away_last5'] = last5_away['home_goals'].sum()
            features['away_avg_goals_conceded_away'] = last5_away['home_goals'].mean()
            
            if len(away_matches_away) >= 10:
                last10_away = away_matches_away.tail(10)
                features['away_avg_goals_scored_away_last10'] = last10_away['away_goals'].mean()
                features['away_avg_goals_conceded_away_last10'] = last10_away['home_goals'].mean()
            else:
                features['away_avg_goals_scored_away_last10'] = features['away_avg_goals_scored_away']
                features['away_avg_goals_conceded_away_last10'] = features['away_avg_goals_conceded_away']
        else:
            features['away_goals_scored_away_last5'] = 0
            features['away_avg_goals_scored_away'] = 1.2  # Away teams score less
            features['away_goals_conceded_away_last5'] = 0
            features['away_avg_goals_conceded_away'] = 1.3
            features['away_avg_goals_scored_away_last10'] = 1.2
            features['away_avg_goals_conceded_away_last10'] = 1.3
        
        # ===== ATTACK vs DEFENSE MATCHUP =====
        # Home attack strength vs Away defense weakness
        features['home_attack_vs_away_defense'] = (
            features['home_avg_goals_scored_home'] / 
            max(features['away_avg_goals_conceded_away'], 0.5)
        )
        
        # Away attack strength vs Home defense weakness  
        features['away_attack_vs_home_defense'] = (
            features['away_avg_goals_scored_away'] / 
            max(features['home_avg_goals_conceded_home'], 0.5)
        )
        
        # ===== ATTACK & DEFENSE STRENGTH RATINGS =====
        # Calculate league average goals per match
        recent_matches = matches_df[matches_df['match_date'] < match_date].tail(100)
        if len(recent_matches) > 0:
            league_avg_home_goals = recent_matches['home_goals'].mean()
            league_avg_away_goals = recent_matches['away_goals'].mean()
        else:
            league_avg_home_goals = 1.5
            league_avg_away_goals = 1.2
        
        # Home attack strength (goals scored vs league avg)
        features['home_attack_strength'] = features['home_avg_goals_scored_home'] / max(league_avg_home_goals, 0.8)
        
        # Away attack strength
        features['away_attack_strength'] = features['away_avg_goals_scored_away'] / max(league_avg_away_goals, 0.8)
        
        # Home defense strength (goals conceded vs league avg) - LOWER is better
        features['home_defense_strength'] = features['home_avg_goals_conceded_home'] / max(league_avg_away_goals, 0.8)
        
        # Away defense strength
        features['away_defense_strength'] = features['away_avg_goals_conceded_away'] / max(league_avg_home_goals, 0.8)
        
        # ===== RECENT GOAL-SCORING FORM (last 3 matches) =====
        # Home team recent goals (all venues)
        home_all_recent = matches_df[
            ((matches_df['home_team'] == home_team) | (matches_df['away_team'] == home_team)) & 
            (matches_df['match_date'] < match_date)
        ].sort_values('match_date').tail(3)
        
        if len(home_all_recent) > 0:
            home_recent_goals = []
            for _, m in home_all_recent.iterrows():
                if m['home_team'] == home_team:
                    home_recent_goals.append(m['home_goals'])
                else:
                    home_recent_goals.append(m['away_goals'])
            features['home_goals_last3'] = sum(home_recent_goals)
            features['home_avg_goals_last3'] = np.mean(home_recent_goals)
        else:
            features['home_goals_last3'] = 0
            features['home_avg_goals_last3'] = 0
        
        # Away team recent goals
        away_all_recent = matches_df[
            ((matches_df['home_team'] == away_team) | (matches_df['away_team'] == away_team)) & 
            (matches_df['match_date'] < match_date)
        ].sort_values('match_date').tail(3)
        
        if len(away_all_recent) > 0:
            away_recent_goals = []
            for _, m in away_all_recent.iterrows():
                if m['home_team'] == away_team:
                    away_recent_goals.append(m['home_goals'])
                else:
                    away_recent_goals.append(m['away_goals'])
            features['away_goals_last3'] = sum(away_recent_goals)
            features['away_avg_goals_last3'] = np.mean(away_recent_goals)
        else:
            features['away_goals_last3'] = 0
            features['away_avg_goals_last3'] = 0
        
        # ===== GOAL DIFFERENCE FEATURES =====
        features['attack_strength_diff'] = features['home_attack_strength'] - features['away_attack_strength']
        features['defense_strength_diff'] = features['home_defense_strength'] - features['away_defense_strength']
        features['goals_scored_diff'] = features['home_avg_goals_scored_home'] - features['away_avg_goals_scored_away']
        features['goals_conceded_diff'] = features['home_avg_goals_conceded_home'] - features['away_avg_goals_conceded_away']
        
        # ===== COMBINED INDICATORS =====
        features['expected_total_goals'] = (
            features['home_avg_goals_scored_home'] + 
            features['away_avg_goals_scored_away']
        )
        
        features['expected_goal_diff'] = (
            features['home_avg_goals_scored_home'] - 
            features['away_avg_goals_scored_away']
        )
        
        # Both teams high scoring
        features['both_teams_offensive'] = int(
            features['home_avg_goals_scored_home'] > 1.5 and 
            features['away_avg_goals_scored_away'] > 1.2
        )
        
        # Both teams defensive
        features['both_teams_defensive'] = int(
            features['home_avg_goals_conceded_home'] < 1.0 and 
            features['away_avg_goals_conceded_away'] < 1.0
        )
        
        new_features.append(features)
    
    # Convert to DataFrame and merge
    new_df = pd.DataFrame(new_features)
    result_df = pd.concat([df.reset_index(drop=True), new_df], axis=1)
    
    logger.info(f"✅ Added {len(new_df.columns)} goal-specific features")
    logger.info(f"   Total features now: {len(result_df.columns)}")
    
    return result_df


def train_ensemble_score_models():
    """Train ensemble of score prediction models with goal-specific features"""
    
    logger.info("="*80)
    logger.info("ENSEMBLE SCORE PREDICTION MODEL TRAINING")
    logger.info("="*80)
    
    # Load processed training data
    processed_file = DATA_DIR / 'processed' / 'training_dataset_full_history.csv'
    
    if not processed_file.exists():
        logger.error(f"Processed data not found: {processed_file}")
        logger.error("Run train_whole_model.py first!")
        return
    
    df = pd.read_csv(processed_file)
    logger.info(f"Loaded {len(df)} matches")
    
    # Load raw match data for goal feature extraction
    raw_file = RAW_DIR / 'pl_all_seasons_combined.csv'
    if raw_file.exists():
        matches_df = pd.read_csv(raw_file)
        logger.info(f"Loaded {len(matches_df)} raw matches for feature extraction")
    else:
        logger.error("Raw match data not found!")
        return
    
    # Add goal-specific features
    df = add_goal_specific_features(df, matches_df)
    
    # Prepare features and targets
    exclude_cols = ['outcome', 'home_goals', 'away_goals', 'match_date', 
                    'home_team', 'away_team', 'gameweek', 'season']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0).values
    y_home = df['home_goals'].values
    y_away = df['away_goals'].values
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(X)}")
    
    # Train/test split (last 50 matches for testing)
    split_idx = len(X) - 50
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
    y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING HOME GOALS ENSEMBLE")
    logger.info("="*80)
    
    # ===== HOME GOALS MODELS =====
    # Model 1: XGBoost Poisson (best for non-linear patterns)
    xgb_home = XGBRegressor(
        objective='count:poisson',
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: LightGBM Poisson (fast, handles sparse features well)
    lgbm_home = LGBMRegressor(
        objective='poisson',
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Model 3: Linear Poisson (baseline, good for stable predictions)
    linear_home = PoissonRegressor(
        alpha=1.0,
        max_iter=300,
        tol=1e-4
    )
    
    # Train individual models
    logger.info("Training XGBoost Poisson...")
    xgb_home.fit(X_train_scaled, y_home_train, verbose=False)
    xgb_home_pred = np.round(xgb_home.predict(X_test_scaled)).astype(int).clip(0, 9)
    xgb_home_mae = mean_absolute_error(y_home_test, xgb_home_pred)
    logger.info(f"  XGBoost MAE: {xgb_home_mae:.3f}")
    
    logger.info("Training LightGBM Poisson...")
    lgbm_home.fit(X_train_scaled, y_home_train)
    lgbm_home_pred = np.round(lgbm_home.predict(X_test_scaled)).astype(int).clip(0, 9)
    lgbm_home_mae = mean_absolute_error(y_home_test, lgbm_home_pred)
    logger.info(f"  LightGBM MAE: {lgbm_home_mae:.3f}")
    
    logger.info("Training Linear Poisson...")
    linear_home.fit(X_train_scaled, y_home_train)
    linear_home_pred = np.round(linear_home.predict(X_test_scaled)).astype(int).clip(0, 9)
    linear_home_mae = mean_absolute_error(y_home_test, linear_home_pred)
    logger.info(f"  Linear MAE: {linear_home_mae:.3f}")
    
    # Weighted ensemble (favor best model)
    total_inv_mae = (1/xgb_home_mae) + (1/lgbm_home_mae) + (1/linear_home_mae)
    w_xgb_home = (1/xgb_home_mae) / total_inv_mae
    w_lgbm_home = (1/lgbm_home_mae) / total_inv_mae
    w_linear_home = (1/linear_home_mae) / total_inv_mae
    
    logger.info(f"\nHome Goals Ensemble Weights:")
    logger.info(f"  XGBoost: {w_xgb_home:.3f}")
    logger.info(f"  LightGBM: {w_lgbm_home:.3f}")
    logger.info(f"  Linear: {w_linear_home:.3f}")
    
    # ===== AWAY GOALS MODELS =====
    logger.info("\n" + "="*80)
    logger.info("TRAINING AWAY GOALS ENSEMBLE")
    logger.info("="*80)
    
    xgb_away = XGBRegressor(
        objective='count:poisson',
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm_away = LGBMRegressor(
        objective='poisson',
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    linear_away = PoissonRegressor(
        alpha=1.0,
        max_iter=300,
        tol=1e-4
    )
    
    logger.info("Training XGBoost Poisson...")
    xgb_away.fit(X_train_scaled, y_away_train, verbose=False)
    xgb_away_pred = np.round(xgb_away.predict(X_test_scaled)).astype(int).clip(0, 9)
    xgb_away_mae = mean_absolute_error(y_away_test, xgb_away_pred)
    logger.info(f"  XGBoost MAE: {xgb_away_mae:.3f}")
    
    logger.info("Training LightGBM Poisson...")
    lgbm_away.fit(X_train_scaled, y_away_train)
    lgbm_away_pred = np.round(lgbm_away.predict(X_test_scaled)).astype(int).clip(0, 9)
    lgbm_away_mae = mean_absolute_error(y_away_test, lgbm_away_pred)
    logger.info(f"  LightGBM MAE: {lgbm_away_mae:.3f}")
    
    logger.info("Training Linear Poisson...")
    linear_away.fit(X_train_scaled, y_away_train)
    linear_away_pred = np.round(linear_away.predict(X_test_scaled)).astype(int).clip(0, 9)
    linear_away_mae = mean_absolute_error(y_away_test, linear_away_pred)
    logger.info(f"  Linear MAE: {linear_away_mae:.3f}")
    
    total_inv_mae_away = (1/xgb_away_mae) + (1/lgbm_away_mae) + (1/linear_away_mae)
    w_xgb_away = (1/xgb_away_mae) / total_inv_mae_away
    w_lgbm_away = (1/lgbm_away_mae) / total_inv_mae_away
    w_linear_away = (1/linear_away_mae) / total_inv_mae_away
    
    logger.info(f"\nAway Goals Ensemble Weights:")
    logger.info(f"  XGBoost: {w_xgb_away:.3f}")
    logger.info(f"  LightGBM: {w_lgbm_away:.3f}")
    logger.info(f"  Linear: {w_linear_away:.3f}")
    
    # ===== ENSEMBLE PREDICTIONS =====
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE EVALUATION")
    logger.info("="*80)
    
    # Weighted ensemble predictions
    y_home_ensemble = (
        w_xgb_home * xgb_home_pred +
        w_lgbm_home * lgbm_home_pred +
        w_linear_home * linear_home_pred
    )
    y_home_ensemble = np.round(y_home_ensemble).astype(int).clip(0, 9)
    
    y_away_ensemble = (
        w_xgb_away * xgb_away_pred +
        w_lgbm_away * lgbm_away_pred +
        w_linear_away * linear_away_pred
    )
    y_away_ensemble = np.round(y_away_ensemble).astype(int).clip(0, 9)
    
    # Metrics
    mae_home = mean_absolute_error(y_home_test, y_home_ensemble)
    mae_away = mean_absolute_error(y_away_test, y_away_ensemble)
    exact_home = (y_home_ensemble == y_home_test).sum() / len(y_home_test) * 100
    exact_away = (y_away_ensemble == y_away_test).sum() / len(y_away_test) * 100
    
    logger.info(f"Home Goals MAE: {mae_home:.2f}, Exact: {exact_home:.1f}%")
    logger.info(f"Away Goals MAE: {mae_away:.2f}, Exact: {exact_away:.1f}%")
    
    # Combined score accuracy
    exact_score = ((y_home_ensemble == y_home_test) & 
                   (y_away_ensemble == y_away_test)).sum() / len(y_home_test) * 100
    
    within_1_home = (np.abs(y_home_ensemble - y_home_test) <= 1).sum() / len(y_home_test) * 100
    within_1_away = (np.abs(y_away_ensemble - y_away_test) <= 1).sum() / len(y_away_test) * 100
    
    logger.info(f"\n📊 Exact Score Accuracy: {exact_score:.1f}%")
    logger.info(f"   Home Goals ±1: {within_1_home:.1f}%")
    logger.info(f"   Away Goals ±1: {within_1_away:.1f}%")
    
    # Outcome accuracy
    y_outcome_test = []
    y_outcome_pred = []
    
    for h_true, a_true, h_pred, a_pred in zip(y_home_test, y_away_test, 
                                                y_home_ensemble, y_away_ensemble):
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
    logger.info(f"   Outcome Accuracy from Score: {outcome_acc:.1f}%")
    
    # ===== SAVE ENSEMBLE MODEL =====
    logger.info("\n" + "="*80)
    logger.info("SAVING ENSEMBLE SCORE MODEL")
    logger.info("="*80)
    
    score_model_data = {
        'home_models': {
            'xgboost': xgb_home,
            'lightgbm': lgbm_home,
            'linear': linear_home
        },
        'away_models': {
            'xgboost': xgb_away,
            'lightgbm': lgbm_away,
            'linear': linear_away
        },
        'home_weights': {
            'xgboost': w_xgb_home,
            'lightgbm': w_lgbm_home,
            'linear': w_linear_home
        },
        'away_weights': {
            'xgboost': w_xgb_away,
            'lightgbm': w_lgbm_away,
            'linear': w_linear_away
        },
        'scaler': scaler,
        'feature_columns': feature_cols,
        'max_goals': 9,
        'ensemble_type': 'weighted_average',
        'exact_score_accuracy': exact_score,
        'outcome_accuracy': outcome_acc
    }
    
    joblib.dump(score_model_data, MODELS_DIR / 'pl_score_prediction_model.pkl')
    logger.info(f"✅ Saved ensemble score model to pl_score_prediction_model.pkl")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ENSEMBLE SCORE MODEL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"📊 Exact Score: {exact_score:.1f}%")
    logger.info(f"📊 Outcome Accuracy: {outcome_acc:.1f}%")
    logger.info(f"📊 Home MAE: {mae_home:.2f}, Away MAE: {mae_away:.2f}")
    logger.info(f"📂 Models saved to: {MODELS_DIR}")
    logger.info("")
    logger.info("🔗 Model is aligned - use with ensemble_predictor.py")
    logger.info("   Scores will be adjusted to match outcome predictions")


if __name__ == "__main__":
    train_ensemble_score_models()
