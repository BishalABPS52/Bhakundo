#!/usr/bin/env python3
"""
Test Retrained Models & Save Predictions to Database
- Populates standings table with GW32 data
- Generates GW33 predictions using retrained models
- Saves predictions to Prediction table in database
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import joblib
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.database import Prediction, Standing, Actual, init_db, get_db_session
from backend.football_api import FootballAPI
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'required' / 'data'
MODELS_DIR = DATA_DIR / 'models'
RAW_DIR = DATA_DIR / 'raw' / 'pl'


class RetractedModelPredictor:
    """Load and use retrained models for predictions"""
    
    def __init__(self):
        """Load all trained models"""
        logger.info("="*80)
        logger.info("LOADING RETRAINED MODELS")
        logger.info("="*80)
        
        # Load ensemble model (XGBoost + LightGBM outcome prediction)
        ensemble_data = joblib.load(MODELS_DIR / 'pl_ensemble_model.pkl')
        self.ensemble_models = ensemble_data['models']  # [xgb, lgbm]
        self.ensemble_weights = ensemble_data['weights']  # Accuracy-based weights
        self.scaler = ensemble_data['scaler']
        self.feature_columns = ensemble_data['feature_columns']
        self.outcome_mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        
        # Load score models (XGBoost + LightGBM regression)
        score_data = joblib.load(MODELS_DIR / 'pl_score_model.pkl')
        self.home_models = score_data['home_models']  # [xgb_home, lgbm_home]
        self.away_models = score_data['away_models']  # [xgb_away, lgbm_away]
        self.mae_home = score_data.get('mae_home', 0.5)
        self.mae_away = score_data.get('mae_away', 0.5)
        
        logger.info(f"✅ Ensemble Model: XGBoost (w={self.ensemble_weights[0]:.3f}) + LightGBM (w={self.ensemble_weights[1]:.3f})")
        logger.info(f"✅ Feature Columns: {len(self.feature_columns)}")
        logger.info(f"✅ Score Models: Home MAE={self.mae_home:.3f}, Away MAE={self.mae_away:.3f}")
    
    def predict_match(self, features):
        """Predict match outcome and scores"""
        # Normalize features
        features_scaled = self.scaler.transform([features])[0]
        
        # Outcome prediction (ensemble)
        xgb_proba = self.ensemble_models[0].predict_proba([features_scaled])[0]
        lgbm_proba = self.ensemble_models[1].predict_proba([features_scaled])[0]
        ensemble_proba = (self.ensemble_weights[0] * xgb_proba + 
                         self.ensemble_weights[1] * lgbm_proba)
        
        # Score prediction (ensemble)
        home_scores_xgb = self.home_models[0].predict([features_scaled])[0]
        home_scores_lgbm = self.home_models[1].predict([features_scaled])[0]
        home_goals = int(round((home_scores_xgb + home_scores_lgbm) / 2))
        
        away_scores_xgb = self.away_models[0].predict([features_scaled])[0]
        away_scores_lgbm = self.away_models[1].predict([features_scaled])[0]
        away_goals = int(round((away_scores_xgb + away_scores_lgbm) / 2))
        
        # Ensure non-negative scores
        home_goals = max(0, home_goals)
        away_goals = max(0, away_goals)
        
        # Determine outcome
        outcome_idx = np.argmax(ensemble_proba)
        outcome = self.outcome_mapping[outcome_idx]
        
        # Confidence score (max probability)
        confidence = float(np.max(ensemble_proba))
        
        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'outcome': outcome,
            'home_prob': float(ensemble_proba[0]),
            'draw_prob': float(ensemble_proba[1]),
            'away_prob': float(ensemble_proba[2]),
            'confidence': confidence
        }


def load_historical_data():
    """Load historical data for feature engineering"""
    logger.info("Loading historical data...")
    
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
            dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['match_id'], keep='last')
    combined_df['match_date'] = pd.to_datetime(combined_df['match_date'])
    combined_df = combined_df.sort_values('match_date').reset_index(drop=True)
    
    logger.info(f"✅ Loaded {len(combined_df)} historical matches")
    return combined_df


def calculate_form_features(team, matches_df, as_of_date, last_n=5):
    """Calculate form features for a team"""
    home = matches_df[(matches_df['home_team'] == team) & 
                      (matches_df['match_date'] < as_of_date)].copy()
    home['goals_for'] = home['home_goals']
    home['goals_against'] = home['away_goals']
    
    away = matches_df[(matches_df['away_team'] == team) & 
                      (matches_df['match_date'] < as_of_date)].copy()
    away['goals_for'] = away['away_goals']
    away['goals_against'] = away['home_goals']
    
    team_matches = pd.concat([home, away]).sort_values('match_date').tail(last_n)
    
    if len(team_matches) == 0:
        return {f'form_last{last_n}': 0, f'ppg_last{last_n}': 0, 
                f'goals_for_last{last_n}': 0, f'goals_against_last{last_n}': 0,
                f'goal_diff_last{last_n}': 0, f'wins_last{last_n}': 0,
                f'draws_last{last_n}': 0, f'losses_last{last_n}': 0, 
                f'clean_sheets_last{last_n}': 0, f'btts_last{last_n}': 0,
                f'over25_last{last_n}': 0, f'avg_goals_for_last{last_n}': 0,
                f'avg_goals_against_last{last_n}': 0}
    
    team_matches['result'] = team_matches.apply(
        lambda x: 'W' if x['goals_for'] > x['goals_against'] 
        else ('D' if x['goals_for'] == x['goals_against'] else 'L'), axis=1)
    team_matches['points'] = team_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
    team_matches['clean_sheet'] = (team_matches['goals_against'] == 0).astype(int)
    team_matches['btts'] = ((team_matches['goals_for'] > 0) & 
                            (team_matches['goals_against'] > 0)).astype(int)
    team_matches['over25'] = ((team_matches['goals_for'] + 
                               team_matches['goals_against']) > 2.5).astype(int)
    
    total_gf = team_matches['goals_for'].sum()
    total_ga = team_matches['goals_against'].sum()
    
    return {f'form_last{last_n}': team_matches['points'].sum(), 
            f'ppg_last{last_n}': team_matches['points'].mean(),
            f'goals_for_last{last_n}': total_gf, 
            f'goals_against_last{last_n}': total_ga,
            f'goal_diff_last{last_n}': total_gf - total_ga, 
            f'wins_last{last_n}': (team_matches['result'] == 'W').sum(),
            f'draws_last{last_n}': (team_matches['result'] == 'D').sum(), 
            f'losses_last{last_n}': (team_matches['result'] == 'L').sum(),
            f'clean_sheets_last{last_n}': team_matches['clean_sheet'].sum(), 
            f'btts_last{last_n}': team_matches['btts'].sum(),
            f'over25_last{last_n}': team_matches['over25'].sum(), 
            f'avg_goals_for_last{last_n}': total_gf / len(team_matches),
            f'avg_goals_against_last{last_n}': total_ga / len(team_matches)}


def build_match_features(home_team, away_team, match_date, historical_df, feature_cols):
    """Build features for a single match"""
    features = {}
    
    # Home team features
    home_form_5 = calculate_form_features(home_team, historical_df, match_date, 5)
    home_form_10 = calculate_form_features(home_team, historical_df, match_date, 10)
    for k, v in {**home_form_5, **home_form_10}.items():
        features[f'home_{k}'] = v
    
    # Away team features
    away_form_5 = calculate_form_features(away_team, historical_df, match_date, 5)
    away_form_10 = calculate_form_features(away_team, historical_df, match_date, 10)
    for k, v in {**away_form_5, **away_form_10}.items():
        features[f'away_{k}'] = v
    
    # Differentials
    for key in home_form_5:
        features[f'diff_{key}'] = features.get(f'home_{key}', 0) - features.get(f'away_{key}', 0)
    
    # Extract only required columns
    feature_vector = [features.get(col, 0) for col in feature_cols]
    return np.array(feature_vector)


def populate_gw32_standings():
    """Populate standings table with GW32 data"""
    logger.info("\n" + "="*80)
    logger.info("POPULATING GW32 STANDINGS")
    logger.info("="*80)
    
    api = FootballAPI()
    standings_data = api.get_standings()
    
    if not standings_data:
        logger.warning("⚠️  Could not fetch standings from API")
        return
    
    db = get_db_session()
    
    try:
        # Clear existing GW32 standings
        db.query(Standing).filter(
            Standing.season == '2025-26',
            Standing.gameweek == 32
        ).delete()
        db.commit()
        
        # Insert new standings
        logger.info(f"💾 Inserting {len(standings_data)} teams for GW32...")
        for team_data in standings_data:
            standing = Standing(
                season='2025-26',
                gameweek=32,
                team=team_data.get('team'),
                position=team_data.get('position'),
                played=team_data.get('played', 0),
                won=team_data.get('won', 0),
                drawn=team_data.get('drawn', 0),
                lost=team_data.get('lost', 0),
                goals_for=team_data.get('goals_for', team_data.get('gf', 0)),
                goals_against=team_data.get('goals_against', team_data.get('ga', 0)),
                goal_difference=team_data.get('goal_difference', team_data.get('gd', 0)),
                points=team_data.get('points', 0),
                form=team_data.get('form', ''),
                updated_at=datetime.utcnow(),
                source='api'
            )
            db.add(standing)
        
        db.commit()
        logger.info(f"✅ Successfully populated {len(standings_data)} teams for GW32")
        
        # Show top 5 teams
        logger.info("\n📊 GW32 TOP 5 STANDINGS:")
        stats = db.query(Standing).filter(
            Standing.season == '2025-26',
            Standing.gameweek == 32
        ).order_by(Standing.position).limit(5).all()
        
        for s in stats:
            logger.info(f"  {s.position}. {s.team:<30} {s.points:>3} pts | {s.played:>2}P {s.won:>2}W {s.drawn:>2}D {s.lost:>2}L | GD: {s.goal_difference:>3}")
        
    except Exception as e:
        logger.error(f"❌ Error populating standings: {e}")
        db.rollback()
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def generate_gw33_predictions():
    """Generate GW33 predictions using retrained models and save to database"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING GW33 PREDICTIONS")
    logger.info("="*80)
    
    # Initialize API and predictor
    api = FootballAPI()
    predictor = RetractedModelPredictor()
    
    # Load historical data for feature engineering
    historical_df = load_historical_data()
    
    # Get GW33 fixtures (both finished and scheduled for comparison)
    logger.info("🔄 Fetching GW33 fixtures...")
    fixtures_scheduled = api.get_matches(status='SCHEDULED', gameweek=33)
    fixtures_completed = api.get_matches(status='FINISHED', gameweek=33)
    fixtures = fixtures_scheduled + fixtures_completed
    
    if not fixtures:
        logger.warning("⚠️  No fixtures found for GW33")
        return
    
    logger.info(f"✅ Retrieved {len(fixtures)} fixtures for GW33")
    
    # Get database session
    db = get_db_session()
    predictions_saved = 0
    
    try:
        for fixture in fixtures:
            try:
                home_team = fixture.get('home_team')
                away_team = fixture.get('away_team')
                match_date = pd.to_datetime(fixture.get('date', datetime.now()), errors='coerce')
                if pd.isna(match_date):
                    match_date = datetime.now()
                match_id = fixture.get('match_id', f"{home_team}_{away_team}_{match_date}")
                
                logger.info(f"\n  Predicting: {home_team} vs {away_team}")
                
                # Check if prediction already exists (will update if it does)
                existing = db.query(Prediction).filter(
                    Prediction.match_id == match_id
                ).first()
                
                # Build features
                feature_vector = build_match_features(
                    home_team, away_team, match_date, 
                    historical_df, predictor.feature_columns
                )
                
                # Make prediction
                pred_result = predictor.predict_match(feature_vector)
                
                logger.info(f"    🎯 Prediction: {home_team} {pred_result['home_goals']}-{pred_result['away_goals']} {away_team}")
                logger.info(f"    📊 Outcome: {pred_result['outcome']} (conf: {pred_result['confidence']:.1%})")
                logger.info(f"    📈 Probabilities: Home {pred_result['home_prob']:.1%} | Draw {pred_result['draw_prob']:.1%} | Away {pred_result['away_prob']:.1%}")
                
                # Save to database (update if exists, insert if new)
                if existing:
                    # Update existing prediction with new values
                    existing.predicted_home_goals = pred_result['home_goals']
                    existing.predicted_away_goals = pred_result['away_goals']
                    existing.predicted_outcome = pred_result['outcome']
                    existing.base_home_prob = pred_result['home_prob']
                    existing.base_draw_prob = pred_result['draw_prob']
                    existing.base_away_prob = pred_result['away_prob']
                    existing.lineup_home_prob = pred_result['home_prob']
                    existing.lineup_draw_prob = pred_result['draw_prob']
                    existing.lineup_away_prob = pred_result['away_prob']
                    existing.ensemble_home_prob = pred_result['home_prob']
                    existing.ensemble_draw_prob = pred_result['draw_prob']
                    existing.ensemble_away_prob = pred_result['away_prob']
                    existing.confidence = pred_result['confidence']
                    existing.ensemble_method = 'XGBoost + LightGBM Weighted Ensemble'
                    existing.verdict = f"{pred_result['outcome']} (Confidence: {pred_result['confidence']:.1%}, Score: {pred_result['home_goals']}-{pred_result['away_goals']})"
                    existing.updated_at = datetime.utcnow()
                    logger.info(f"    ♻️  Updated existing prediction")
                else:
                    # Create new prediction
                    prediction = Prediction(
                        match_id=str(match_id),
                        home_team=home_team,
                        away_team=away_team,
                        gameweek=33,
                        match_date=match_date,
                        season='2025-26',
                        
                        # Predicted scores
                        predicted_home_goals=pred_result['home_goals'],
                        predicted_away_goals=pred_result['away_goals'],
                        predicted_outcome=pred_result['outcome'],
                        
                        # Base model probabilities (using ensemble)
                        base_home_prob=pred_result['home_prob'],
                        base_draw_prob=pred_result['draw_prob'],
                        base_away_prob=pred_result['away_prob'],
                        
                        # Lineup model probabilities (using same for now)
                        lineup_home_prob=pred_result['home_prob'],
                        lineup_draw_prob=pred_result['draw_prob'],
                        lineup_away_prob=pred_result['away_prob'],
                        
                        # Ensemble probabilities
                        ensemble_home_prob=pred_result['home_prob'],
                        ensemble_draw_prob=pred_result['draw_prob'],
                        ensemble_away_prob=pred_result['away_prob'],
                        
                        # Confidence and verdict
                        confidence=pred_result['confidence'],
                        ensemble_method='XGBoost + LightGBM Weighted Ensemble',
                        verdict=f"{pred_result['outcome']} (Confidence: {pred_result['confidence']:.1%}, Score: {pred_result['home_goals']}-{pred_result['away_goals']})",
                        
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    db.add(prediction)
                    logger.info(f"    ✅ Created new prediction")
                
                db.commit()
                predictions_saved += 1
                
            except Exception as e:
                logger.error(f"    ❌ Error processing {home_team} vs {away_team}: {e}")
                db.rollback()
                continue
    
    finally:
        db.close()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ GW33 PREDICTIONS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Predictions generated: {predictions_saved}/{len(fixtures)}")
    logger.info(f"📍 All predictions saved to 'prediction' table in database")
    logger.info(f"{'='*80}\n")


def main():
    """Main execution"""
    try:
        # Initialize database
        init_db()
        logger.info("✅ Database initialized")
        
        # Step 1: Populate GW32 standings
        populate_gw32_standings()
        
        # Step 2: Generate GW33 predictions
        generate_gw33_predictions()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 COMPLETE! GW32 standings populated and GW33 predictions saved to database")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
