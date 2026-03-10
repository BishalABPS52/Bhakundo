"""Lineup prediction models and utilities"""

import numpy as np
import pandas as pd
from typing import Dict


class LineupPredictor:
    """Predicts team lineups based on recent form and team patterns"""
    
    def __init__(self):
        self.formations = {
            'defensive': ['4-5-1', '5-4-1', '5-3-2'],
            'balanced': ['4-4-2', '4-3-3', '4-2-3-1'],
            'attacking': ['3-4-3', '4-3-3', '3-5-2']
        }
        
        # Common Premier League formations
        self.common_formations = ['4-3-3', '4-2-3-1', '3-4-3', '4-4-2', '3-5-2', '5-3-2']
        
    def predict_formation(self, team_features: Dict) -> str:
        """Predict formation based on team strategy and opponent"""
        
        # Use team's recent form and playing style
        ppg = team_features.get('ppg_last5', 1.0)
        goals_for = team_features.get('goals_for_last5', 5)
        goals_against = team_features.get('goals_against_last5', 5)
        
        # Determine playing style
        if goals_for > 8 and ppg > 2.0:
            style = 'attacking'
        elif goals_against < 4 and ppg < 1.5:
            style = 'defensive'
        else:
            style = 'balanced'
        
        # Select formation based on style
        formations = self.formations[style]
        
        # Add some randomness (teams don't always use same formation)
        if np.random.random() < 0.3:  # 30% chance of variation
            return np.random.choice(self.common_formations)
        
        return np.random.choice(formations)
    
    def encode_formation(self, formation: str) -> Dict:
        """Encode formation into defensive/midfield/attacking strength"""
        
        if not formation or formation == 'Unknown':
            return {'def_line': 4, 'mid_line': 4, 'att_line': 2}
        
        parts = formation.split('-')
        
        if len(parts) == 3:
            return {
                'def_line': int(parts[0]),
                'mid_line': int(parts[1]),
                'att_line': int(parts[2])
            }
        else:
            return {'def_line': 4, 'mid_line': 4, 'att_line': 2}


class LineupResultModel:
    """Predicts match results using lineup and formation features"""
    
    def __init__(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.scaler = StandardScaler()
        self.formation_encoder = LabelEncoder()
        self.models = {}
        
    def create_lineup_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """Create features that simulate lineup quality and tactical matchups"""
        
        features_list = []
        predictor = LineupPredictor()
        
        for idx, row in match_df.iterrows():
            features = {}
            
            # Basic match info
            features['match_date'] = row['match_date']
            features['home_team'] = row['home_team']
            features['away_team'] = row['away_team']
            
            # Get team form features (from existing dataset)
            home_features = {
                'ppg_last5': row.get('home_ppg_last5', 1.5),
                'goals_for_last5': row.get('home_goals_for_last5', 5),
                'goals_against_last5': row.get('home_goals_against_last5', 5)
            }
            
            away_features = {
                'ppg_last5': row.get('away_ppg_last5', 1.5),
                'goals_for_last5': row.get('away_goals_for_last5', 5),
                'goals_against_last5': row.get('away_goals_against_last5', 5)
            }
            
            # Predict formations
            home_formation = predictor.predict_formation(home_features)
            away_formation = predictor.predict_formation(away_features)
            
            features['home_formation'] = home_formation
            features['away_formation'] = away_formation
            
            # Encode formations
            home_encoded = predictor.encode_formation(home_formation)
            away_encoded = predictor.encode_formation(away_formation)
            
            features['home_def_line'] = home_encoded['def_line']
            features['home_mid_line'] = home_encoded['mid_line']
            features['home_att_line'] = home_encoded['att_line']
            
            features['away_def_line'] = away_encoded['def_line']
            features['away_mid_line'] = away_encoded['mid_line']
            features['away_att_line'] = away_encoded['att_line']
            
            # Tactical matchup features
            features['def_vs_att_home'] = home_encoded['def_line'] - away_encoded['att_line']
            features['def_vs_att_away'] = away_encoded['def_line'] - home_encoded['att_line']
            features['mid_battle'] = home_encoded['mid_line'] - away_encoded['mid_line']
            
            # Formation aggression scores
            features['home_aggression'] = home_encoded['att_line'] + home_encoded['mid_line'] - home_encoded['def_line']
            features['away_aggression'] = away_encoded['att_line'] + away_encoded['mid_line'] - away_encoded['def_line']
            
            # Copy all existing features
            for col in match_df.columns:
                if col not in features and col not in ['outcome', 'home_goals', 'away_goals', 'goal_diff', 'season', 'gameweek']:
                    features[col] = row[col]
            
            # Target
            if 'outcome' in row:
                features['outcome'] = row['outcome']
            if 'home_goals' in row:
                features['home_goals'] = row['home_goals']
            if 'away_goals' in row:
                features['away_goals'] = row['away_goals']
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train ensemble models with lineup features"""
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("Training XGBoost with lineup features...")
        self.models['xgboost'] = XGBClassifier(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=9,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.05,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        self.models['xgboost'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info("Training LightGBM with lineup features...")
        self.models['lightgbm'] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=60,
            max_depth=10,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            force_row_wise=True
        )
        self.models['lightgbm'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
        
        logger.info("Training CatBoost with lineup features...")
        self.models['catboost'] = CatBoostClassifier(
            iterations=800,
            learning_rate=0.02,
            depth=9,
            l2_leaf_reg=2,
            random_strength=0.3,
            bagging_temperature=0.1,
            random_state=42,
            verbose=False,
            thread_count=-1
        )
        self.models['catboost'].fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
    
    def predict_proba(self, X):
        """Ensemble prediction with averaged probabilities"""
        probas = []
        weights = [0.30, 0.35, 0.35]  # XGB, LGBM, CB
        
        for model_name, weight in zip(['xgboost', 'lightgbm', 'catboost'], weights):
            proba = self.models[model_name].predict_proba(X)
            probas.append(proba * weight)
        
        return np.sum(probas, axis=0)
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
