"""Prediction System - Real-time match predictions"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

from src.features.feature_engineering import FeatureEngineer
from src.models.outcome_model import OutcomeModel
from src.models.score_model import ScorePredictionModel
from src.data.database import get_session, Team, Fixture
import pickle
from src.config import MODELS_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class PredictionSystem:
    """Real-time prediction system for Premier League matches"""
    
    def __init__(self):
        self.engineer = FeatureEngineer()
        self.outcome_model = OutcomeModel()
        self.score_model = ScorePredictionModel()
        self.session = get_session()
        self._matches_df_cache = None
        self.xgb_score_model = None

        # Load trained models
        self._load_models()
        self._load_xgb_score_model()
    
    def _load_models(self):
        """Load trained models"""
        model_path = MODELS_DIR / "outcome_model.pkl"
        
        if model_path.exists():
            self.outcome_model.load_model(model_path)
            logger.info("Outcome model loaded successfully")
        else:
            logger.warning(f"Outcome model not found at {model_path}. Please train models first.")
    
    def _get_matches_df(self) -> pd.DataFrame:
        """Load and cache historical matches from the combined CSV."""
        if self._matches_df_cache is None:
            csv_path = RAW_DATA_DIR / 'pl' / 'pl_all_seasons_combined.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=['match_date'])
                df = df[df['status'] == 'FINISHED'].copy()
                self._matches_df_cache = df
                logger.info(f"Loaded {len(df)} historical matches for dynamic xG")
            else:
                logger.warning(f"Historical CSV not found at {csv_path}")
                self._matches_df_cache = pd.DataFrame()
        return self._matches_df_cache

    def _load_xgb_score_model(self):
        """Load the enhanced LightGBM/XGBoost score blend model if available."""
        model_path = MODELS_DIR / 'score_prediction_model_enhanced_v3.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self.xgb_score_model = pickle.load(f)
                logger.info("Enhanced score blend model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load enhanced score model: {e}")
        else:
            logger.info("Enhanced score model not yet available — using Poisson only")

    def _get_dynamic_xg(self, home_team_name: str, away_team_name: str,
                         match_date) -> tuple:
        """
        Compute dynamic xG from rolling attack/defence strength ratings.
        Uses the last 10 venue-specific matches before match_date.
        Falls back to PL baseline averages if insufficient data.
        """
        df = self._get_matches_df()
        if df.empty:
            return 1.5, 1.2

        PL_HOME = 1.53
        PL_AWAY = 1.22
        W = 10
        cutoff = pd.Timestamp(match_date)

        def _strength(team, is_home):
            if is_home:
                rows = df[(df['home_team'] == team) & (df['match_date'] < cutoff)]
                scored  = rows['home_goals'].tail(W)
                concede = rows['away_goals'].tail(W)
                baseline_atk, baseline_dfe = PL_HOME, PL_AWAY
            else:
                rows = df[(df['away_team'] == team) & (df['match_date'] < cutoff)]
                scored  = rows['away_goals'].tail(W)
                concede = rows['home_goals'].tail(W)
                baseline_atk, baseline_dfe = PL_AWAY, PL_HOME
            if len(scored) == 0:
                return 1.0, 1.0
            atk = float(scored.mean()) / baseline_atk
            dfe = float(concede.mean()) / baseline_dfe
            return atk, dfe

        h_atk, h_dfe = _strength(home_team_name, True)
        a_atk, a_dfe = _strength(away_team_name, False)

        home_xg = float(np.clip(PL_HOME * h_atk * a_dfe, 0.5, 4.0))
        away_xg = float(np.clip(PL_AWAY * a_atk * h_dfe, 0.3, 3.5))
        return home_xg, away_xg

    def predict_match(self, home_team_id: int, away_team_id: int, 
                     match_date: Optional[datetime] = None) -> Dict:
        """
        Generate complete prediction for a single match
        
        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID
            match_date: Date of match (defaults to today)
            
        Returns:
            Dictionary with complete prediction
        """
        if match_date is None:
            match_date = datetime.now()
        
        logger.info(f"Predicting match: {home_team_id} vs {away_team_id} on {match_date}")
        
        try:
            # Get team names
            home_team = self.session.query(Team).filter_by(team_id=home_team_id).first()
            away_team = self.session.query(Team).filter_by(team_id=away_team_id).first()
            
            if not home_team or not away_team:
                raise ValueError("Invalid team IDs")
            
            # Generate features
            features = self.engineer.create_match_features(
                home_team_id, away_team_id, match_date
            )
            
            if features.empty:
                raise ValueError("Could not generate features")
            
            # Remove non-feature columns
            feature_cols = [col for col in features.columns 
                          if col not in ['home_team_id', 'away_team_id', 'match_date']]
            X = features[feature_cols]
            
            # Predict outcome
            if self.outcome_model.is_trained:
                probabilities = self.outcome_model.predict_proba(X)[0]
                
                # Map probabilities to outcomes (assuming label order is A, D, H)
                outcome_probs = {
                    'away_win': float(probabilities[0]),
                    'draw': float(probabilities[1]),
                    'home_win': float(probabilities[2])
                }
                
                # Predicted outcome
                predicted_outcome_idx = np.argmax(probabilities)
                outcome_labels = ['A', 'D', 'H']
                predicted_outcome = outcome_labels[predicted_outcome_idx]
            else:
                outcome_probs = {'home_win': 0.33, 'draw': 0.33, 'away_win': 0.34}
                predicted_outcome = 'Unknown'
            
            # --- Dynamic xG from rolling attack/defence strength ---
            home_xg, away_xg = self._get_dynamic_xg(
                home_team.team_name, away_team.team_name, match_date
            )

            # --- ML + Poisson blend with outcome constraint ---
            outcome_map = {'H': 'home_win', 'D': 'draw', 'A': 'away_win', 'Unknown': None}
            outcome_label = outcome_map.get(predicted_outcome, None)

            # Try to blend with enhanced ML score model
            home_xg_final, away_xg_final = home_xg, away_xg
            if self.xgb_score_model is not None:
                try:
                    scaler = self.xgb_score_model.get('scaler')
                    lgb_h  = self.xgb_score_model.get('lgb_home')
                    lgb_a  = self.xgb_score_model.get('lgb_away')
                    xgb_h  = self.xgb_score_model.get('xgb_home')
                    xgb_a  = self.xgb_score_model.get('xgb_away')
                    if all(m is not None for m in [scaler, lgb_h, lgb_a, xgb_h, xgb_a]):
                        Xs = scaler.transform(X.values)
                        ml_hg = 0.55 * lgb_h.predict(Xs)[0] + 0.45 * xgb_h.predict(Xs)[0]
                        ml_ag = 0.55 * lgb_a.predict(Xs)[0] + 0.45 * xgb_a.predict(Xs)[0]
                        home_xg_final = float(np.clip(ml_hg * 0.60 + home_xg * 0.40, 0.3, 5.0))
                        away_xg_final = float(np.clip(ml_ag * 0.60 + away_xg * 0.40, 0.2, 4.5))
                except Exception as _e:
                    logger.debug(f"Score model blend fallback to Poisson: {_e}")

            home_goals, away_goals, score_prob = self.score_model.predict_most_likely_score(
                home_xg_final, away_xg_final, predicted_outcome=outcome_label
            )
            
            top_scores = self.score_model.predict_top_n_scores(home_xg_final, away_xg_final, n=5)

            # Additional predictions
            over_under = self.score_model.predict_over_under(home_xg_final, away_xg_final, 2.5)
            btts = self.score_model.predict_both_teams_to_score(home_xg_final, away_xg_final)
            
            # Calculate confidence
            max_prob = max(outcome_probs.values())
            if max_prob > 0.55:
                confidence = "High"
            elif max_prob > 0.45:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Compile prediction
            prediction = {
                'match_info': {
                    'home_team': home_team.team_name,
                    'away_team': away_team.team_name,
                    'match_date': match_date.isoformat() if match_date else None,
                },
                'outcome_prediction': {
                    'home_win_probability': outcome_probs['home_win'],
                    'draw_probability': outcome_probs['draw'],
                    'away_win_probability': outcome_probs['away_win'],
                    'predicted_outcome': predicted_outcome,
                    'confidence': confidence,
                },
                'score_prediction': {
                    'most_likely_score': f"{home_goals}-{away_goals}",
                    'home_goals': int(home_goals),
                    'away_goals': int(away_goals),
                    'probability': float(score_prob),
                    'top_5_scorelines': top_scores.to_dict('records')
                },
                'additional_predictions': {
                    'over_2_5': float(over_under['over_2.5']),
                    'under_2_5': float(over_under['under_2.5']),
                    'btts_yes': float(btts['btts_yes']),
                    'btts_no': float(btts['btts_no']),
                },
                'key_factors': {
                    'home_recent_form': features.get('home_last_5_form', ['']).iloc[0],
                    'away_recent_form': features.get('away_last_5_form', ['']).iloc[0],
                    'home_xg_avg': float(home_xg_final),
                    'away_xg_avg': float(away_xg_final),
                }
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            raise
    
    def predict_upcoming_fixtures(self, limit: int = 10) -> List[Dict]:
        """
        Generate predictions for upcoming fixtures
        
        Args:
            limit: Maximum number of fixtures to predict
        """
        logger.info(f"Predicting upcoming {limit} fixtures...")
        
        # Get upcoming fixtures from database
        fixtures = self.session.query(Fixture).filter(
            Fixture.match_date > datetime.now()
        ).order_by(Fixture.match_date).limit(limit).all()
        
        if not fixtures:
            logger.warning("No upcoming fixtures found in database")
            return []
        
        predictions = []
        
        for fixture in fixtures:
            try:
                prediction = self.predict_match(
                    fixture.home_team_id,
                    fixture.away_team_id,
                    fixture.match_date
                )
                prediction['fixture_id'] = fixture.fixture_id
                prediction['gameweek'] = fixture.gameweek
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting fixture {fixture.fixture_id}: {e}")
                continue
        
        logger.info(f"Generated predictions for {len(predictions)} fixtures")
        return predictions
    
    def __del__(self):
        """Close database session"""
        if hasattr(self, 'session'):
            self.session.close()


if __name__ == "__main__":
    # Test prediction system
    logging.basicConfig(level=logging.INFO)
    
    system = PredictionSystem()
    
    # Example: Predict a match (using team IDs from database)
    # This would need actual team IDs from your database
    try:
        prediction = system.predict_match(home_team_id=1, away_team_id=2)
        
        print("\n" + "="*60)
        print("MATCH PREDICTION")
        print("="*60)
        
        import json
        print(json.dumps(prediction, indent=2))
        
    except Exception as e:
        logger.error(f"Could not generate prediction: {e}")
