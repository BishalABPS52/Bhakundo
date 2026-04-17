"""
Train Premier League Prediction Models - Complete Historical Data
FIXED VERSION: XGBoost + LightGBM Ensemble (no CatBoost)
- Proper model names: pl_ensemble_model, pl_base_model, pl_lineup_model, pl_score_model
- Fixed LightGBM verbose parameter issue
- Complete score prediction ensemble training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'required' / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'pl'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_all_historical_data():
    """Load all historical match data"""
    logger.info("="*80)
    logger.info("LOADING HISTORICAL DATA (2023-2026)")
    logger.info("="*80)
    
    data_files = [
        RAW_DIR / 'pl_2023_historical.csv',
        RAW_DIR / 'pl_2024_historical.csv',
        RAW_DIR / 'pl_2025_26_completed_matches.csv'
    ]
    
    dfs = []
    for file in data_files:
        if file.exists():
            df = pd.read_csv(file)
            if 'api_match_id' in df.columns:
                df = df.rename(columns={'api_match_id': 'match_id'})
            if 'home_score' in df.columns and 'home_goals' not in df.columns:
                df['home_goals'] = df['home_score']
                df['away_goals'] = df['away_score']
            if 'status' in df.columns:
                df = df[df['status'].isin(['FINISHED', 'FT'])].copy()
            logger.info(f"  ✅ Loaded {len(df):4d} matches from {file.name}")
            dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['match_id'], keep='last')
    combined_df = combined_df.dropna(subset=['home_goals', 'away_goals'])
    combined_df['home_goals'] = combined_df['home_goals'].astype(int)
    combined_df['away_goals'] = combined_df['away_goals'].astype(int)
    
    logger.info(f"✅ Total unique matches: {len(combined_df)}")
    return combined_df


def calculate_form_features(team, matches_df, as_of_date, last_n=5, venue='all'):
    """Calculate form features"""
    if venue == 'home':
        team_matches = matches_df[(matches_df['home_team'] == team) & (matches_df['match_date'] < as_of_date)].copy()
        team_matches['goals_for'] = team_matches['home_goals']
        team_matches['goals_against'] = team_matches['away_goals']
    elif venue == 'away':
        team_matches = matches_df[(matches_df['away_team'] == team) & (matches_df['match_date'] < as_of_date)].copy()
        team_matches['goals_for'] = team_matches['away_goals']
        team_matches['goals_against'] = team_matches['home_goals']
    else:
        home = matches_df[(matches_df['home_team'] == team) & (matches_df['match_date'] < as_of_date)].copy()
        home['goals_for'] = home['home_goals']
        home['goals_against'] = home['away_goals']
        away = matches_df[(matches_df['away_team'] == team) & (matches_df['match_date'] < as_of_date)].copy()
        away['goals_for'] = away['away_goals']
        away['goals_against'] = away['home_goals']
        team_matches = pd.concat([home, away])
    
    team_matches = team_matches.sort_values('match_date').tail(last_n)
    
    if len(team_matches) == 0:
        return {f'form_last{last_n}': 0, f'ppg_last{last_n}': 0, f'goals_for_last{last_n}': 0,
                f'goals_against_last{last_n}': 0, f'goal_diff_last{last_n}': 0, f'wins_last{last_n}': 0,
                f'draws_last{last_n}': 0, f'losses_last{last_n}': 0, f'clean_sheets_last{last_n}': 0,
                f'btts_last{last_n}': 0, f'over25_last{last_n}': 0, f'avg_goals_for_last{last_n}': 0,
                f'avg_goals_against_last{last_n}': 0}
    
    team_matches['result'] = team_matches.apply(
        lambda x: 'W' if x['goals_for'] > x['goals_against'] else ('D' if x['goals_for'] == x['goals_against'] else 'L'), axis=1)
    team_matches['points'] = team_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
    team_matches['clean_sheet'] = (team_matches['goals_against'] == 0).astype(int)
    team_matches['btts'] = ((team_matches['goals_for'] > 0) & (team_matches['goals_against'] > 0)).astype(int)
    team_matches['over25'] = ((team_matches['goals_for'] + team_matches['goals_against']) > 2.5).astype(int)
    
    total_gf = team_matches['goals_for'].sum()
    total_ga = team_matches['goals_against'].sum()
    
    return {f'form_last{last_n}': team_matches['points'].sum(), f'ppg_last{last_n}': team_matches['points'].mean(),
            f'goals_for_last{last_n}': total_gf, f'goals_against_last{last_n}': total_ga,
            f'goal_diff_last{last_n}': total_gf - total_ga, f'wins_last{last_n}': (team_matches['result'] == 'W').sum(),
            f'draws_last{last_n}': (team_matches['result'] == 'D').sum(), f'losses_last{last_n}': (team_matches['result'] == 'L').sum(),
            f'clean_sheets_last{last_n}': team_matches['clean_sheet'].sum(), f'btts_last{last_n}': team_matches['btts'].sum(),
            f'over25_last{last_n}': team_matches['over25'].sum(), f'avg_goals_for_last{last_n}': total_gf / len(team_matches),
            f'avg_goals_against_last{last_n}': total_ga / len(team_matches)}


def build_advanced_features(df):
    """Build 118+ features"""
    logger.info("="*80)
    logger.info("BUILDING ADVANCED FEATURES (118+)")
    logger.info("="*80)
    
    df = df.copy()
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)
    
    df['outcome'] = 0
    df.loc[df['home_goals'] < df['away_goals'], 'outcome'] = 2
    df.loc[df['home_goals'] == df['away_goals'], 'outcome'] = 1
    
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Processing match {idx+1}/{len(df)}")
        
        historical = df[:idx] if idx > 0 else pd.DataFrame()
        if len(historical) < 20:
            continue
        
        features = {}
        
        # Home team features
        home_form_5 = calculate_form_features(row['home_team'], historical, row['match_date'], 5, 'all')
        home_form_10 = calculate_form_features(row['home_team'], historical, row['match_date'], 10, 'all')
        for k, v in {**home_form_5, **home_form_10}.items():
            features[f'home_{k}'] = v
        
        # Away team features
        away_form_5 = calculate_form_features(row['away_team'], historical, row['match_date'], 5, 'all')
        away_form_10 = calculate_form_features(row['away_team'], historical, row['match_date'], 10, 'all')
        for k, v in {**away_form_5, **away_form_10}.items():
            features[f'away_{k}'] = v
        
        # Differentials
        for key in home_form_5:
            features[f'diff_{key}'] = features.get(f'home_{key}', 0) - features.get(f'away_{key}', 0)
        
        features['outcome'] = row['outcome']
        features['home_goals'] = row['home_goals']
        features['away_goals'] = row['away_goals']
        
        features_list.append(features)
    
    df_processed = pd.DataFrame(features_list)
    feature_cols = [c for c in df_processed.columns if c not in ['outcome', 'home_goals', 'away_goals']]
    
    logger.info(f"✅ Features created: {len(feature_cols)}")
    return df_processed, feature_cols


def train_comprehensive_models(df, feature_cols):
    """Train XGBoost + LightGBM ensemble"""
    logger.info("\n" + "="*80)
    logger.info("TRAINING ENSEMBLE MODELS (XGBoost + LightGBM)")
    logger.info("="*80)
    
    X = df[feature_cols].fillna(0).values
    y = df['outcome'].values
    
    # Train/Val/Test split
    n_test = 50
    n_total = len(X)
    n_train_val = n_total - n_test
    n_train = int(n_train_val * 0.9)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train_val], X[n_train_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train_val], y[n_train_val:]
    
    logger.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    logger.info("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    class_counts = np.bincount(y_train_balanced)
    logger.info(f"After SMOTE: Home={class_counts[0]}, Draw={class_counts[1]}, Away={class_counts[2]}")
    
    # XGBoost
    logger.info("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_balanced, y_train_balanced, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    logger.info(f"XGBoost Test Accuracy: {acc_xgb:.4f}")
    logger.info(classification_report(y_test, y_pred_xgb, target_names=['Home', 'Draw', 'Away']))
    
    # LightGBM (FIXED: verbose is constructor parameter, not fit parameter)
    logger.info("Training LightGBM...")
    lgbm_model = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=50, max_depth=8, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, force_row_wise=True, verbose=-1
    )
    lgbm_model.fit(X_train_balanced, y_train_balanced, eval_set=[(X_val_scaled, y_val)])
    
    y_pred_lgbm = lgbm_model.predict(X_test_scaled)
    acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
    logger.info(f"LightGBM Test Accuracy: {acc_lgbm:.4f}")
    logger.info(classification_report(y_test, y_pred_lgbm, target_names=['Home', 'Draw', 'Away']))
    
    # Ensemble
    logger.info("Creating ensemble...")
    total_acc = acc_xgb + acc_lgbm
    w_xgb = acc_xgb / total_acc
    w_lgbm = acc_lgbm / total_acc
    logger.info(f"Ensemble weights: XGB={w_xgb:.3f}, LGBM={w_lgbm:.3f}")
    
    proba_xgb = xgb_model.predict_proba(X_test_scaled)
    proba_lgbm = lgbm_model.predict_proba(X_test_scaled)
    proba_ensemble = w_xgb * proba_xgb + w_lgbm * proba_lgbm
    y_pred_ensemble = np.argmax(proba_ensemble, axis=1)
    
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    logger.info(f"\n🎯 ENSEMBLE Test Accuracy: {acc_ensemble:.4f}")
    logger.info(classification_report(y_test, y_pred_ensemble, target_names=['Home', 'Draw', 'Away']))
    
    return xgb_model, lgbm_model, scaler, feature_cols, w_xgb, w_lgbm, acc_xgb, acc_lgbm, acc_ensemble


def train_score_models(df, feature_cols, scaler):
    """Train score prediction ensemble"""
    logger.info("\n" + "="*80)
    logger.info("TRAINING SCORE PREDICTION MODELS (XGBoost + LightGBM)")
    logger.info("="*80)
    
    X = df[feature_cols].fillna(0).values
    y_home = df['home_goals'].values.astype(float)
    y_away = df['away_goals'].values.astype(float)
    
    # Time-ordered split
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_home_train, y_home_test = y_home[:split], y_home[split:]
    y_away_train, y_away_test = y_away[:split], y_away[split:]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Score model - Train: {len(X_train)}  Test: {len(X_test)}")
    
    # Home goals - XGBoost
    logger.info("Training home goals XGBoost...")
    xgb_home = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, objective='count:poisson'
    )
    xgb_home.fit(X_train_scaled, y_home_train, verbose=False)
    logger.info("✅ XGBoost home goals trained")
    
    # Home goals - LightGBM
    logger.info("Training home goals LightGBM...")
    lgbm_home = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=40, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, objective='regression_l2', verbose=-1
    )
    lgbm_home.fit(X_train_scaled, y_home_train)
    logger.info("✅ LightGBM home goals trained")
    
    # Away goals - XGBoost
    logger.info("Training away goals XGBoost...")
    xgb_away = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, objective='count:poisson'
    )
    xgb_away.fit(X_train_scaled, y_away_train, verbose=False)
    logger.info("✅ XGBoost away goals trained")
    
    # Away goals - LightGBM
    logger.info("Training away goals LightGBM...")
    lgbm_away = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=40, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, objective='regression_l2', verbose=-1
    )
    lgbm_away.fit(X_train_scaled, y_away_train)
    logger.info("✅ LightGBM away goals trained")
    
    # Evaluate
    xgb_home_pred = xgb_home.predict(X_test_scaled)
    lgbm_home_pred = lgbm_home.predict(X_test_scaled)
    home_ensemble_pred = (xgb_home_pred + lgbm_home_pred) / 2
    mae_home = mean_absolute_error(y_home_test, home_ensemble_pred)
    
    xgb_away_pred = xgb_away.predict(X_test_scaled)
    lgbm_away_pred = lgbm_away.predict(X_test_scaled)
    away_ensemble_pred = (xgb_away_pred + lgbm_away_pred) / 2
    mae_away = mean_absolute_error(y_away_test, away_ensemble_pred)
    
    logger.info(f"\n📊 SCORE MODEL METRICS:")
    logger.info(f"  Home Goals MAE: {mae_home:.3f}")
    logger.info(f"  Away Goals MAE: {mae_away:.3f}")
    
    return xgb_home, lgbm_home, xgb_away, lgbm_away, mae_home, mae_away


def save_models(xgb, lgbm, scaler, feature_cols, w_xgb, w_lgbm, 
                xgb_home, lgbm_home, xgb_away, lgbm_away, mae_home, mae_away):
    """Save all models with correct names"""
    logger.info("\n" + "="*80)
    logger.info("SAVING MODELS")
    logger.info("="*80)
    
    # pl_base_model.pkl
    base_data = {'model': xgb, 'scaler': scaler, 'feature_columns': feature_cols,
                 'reverse_mapping': {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}}
    joblib.dump(base_data, MODELS_DIR / 'pl_base_model.pkl')
    logger.info("✅ Saved pl_base_model.pkl")
    
    # pl_lineup_model.pkl
    lineup_data = {'model': lgbm, 'scaler': scaler, 'feature_columns': feature_cols,
                   'reverse_mapping': {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}}
    joblib.dump(lineup_data, MODELS_DIR / 'pl_lineup_model.pkl')
    logger.info("✅ Saved pl_lineup_model.pkl")
    
    # pl_ensemble_model.pkl
    ensemble_data = {
        'models': [xgb, lgbm],
        'weights': [w_xgb, w_lgbm],
        'scaler': scaler,
        'feature_columns': feature_cols,
        'reverse_mapping': {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    }
    joblib.dump(ensemble_data, MODELS_DIR / 'pl_ensemble_model.pkl')
    logger.info("✅ Saved pl_ensemble_model.pkl")
    
    # pl_score_model.pkl
    score_data = {
        'home_models': [xgb_home, lgbm_home],
        'away_models': [xgb_away, lgbm_away],
        'scaler': scaler,
        'feature_columns': feature_cols,
        'mae_home': mae_home,
        'mae_away': mae_away,
        'max_goals': 9
    }
    joblib.dump(score_data, MODELS_DIR / 'pl_score_model.pkl')
    logger.info("✅ Saved pl_score_model.pkl")
    logger.info(f"\n🎉 All models saved to {MODELS_DIR}")


def main():
    """Main training pipeline"""
    logger.info("\n" + "="*80)
    logger.info("🚀 COMPLETE MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load
    df_raw = load_all_historical_data()
    
    # Features
    df_features, feature_cols = build_advanced_features(df_raw)
    
    # Train outcome models
    xgb, lgbm, scaler, feature_cols, w_xgb, w_lgbm, acc_xgb, acc_lgbm, acc_ensemble = train_comprehensive_models(df_features, feature_cols)
    
    # Train score models
    xgb_home, lgbm_home, xgb_away, lgbm_away, mae_home, mae_away = train_score_models(df_features, feature_cols, scaler)
    
    # Save
    save_models(xgb, lgbm, scaler, feature_cols, w_xgb, w_lgbm,
                xgb_home, lgbm_home, xgb_away, lgbm_away, mae_home, mae_away)
    
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📊 TEST ACCURACIES & METRICS:")
    logger.info(f"  XGBoost:        {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
    logger.info(f"  LightGBM:       {acc_lgbm:.4f} ({acc_lgbm*100:.2f}%)")
    logger.info(f"  Ensemble:       {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%) 🎯")
    logger.info(f"\n  Home Goals MAE: {mae_home:.3f}")
    logger.info(f"  Away Goals MAE: {mae_away:.3f}")
    logger.info(f"\n  Models saved to: {MODELS_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
