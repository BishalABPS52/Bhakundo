"""Master feature engineering module - Combines all features"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List
import logging

from src.features.form_features import FormFeatures
from src.features.xg_features import XGFeatures
from src.data.database import get_session, Match, Team
from src.config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Master feature engineering class"""
    
    def __init__(self):
        self.form_features = FormFeatures()
        self.xg_features = XGFeatures()
        self.session = get_session()
    
    def create_match_features(self, home_team_id: int, away_team_id: int,
                              match_date: datetime) -> pd.DataFrame:
        """
        Create complete feature set for a single match
        
        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID
            match_date: Date of the match
            
        Returns:
            DataFrame with single row of features
        """
        logger.info(f"Creating features for match: Team {home_team_id} vs {away_team_id}")
        
        # Get historical matches
        matches_df = self._get_historical_matches(as_of_date=match_date)
        
        if matches_df.empty:
            logger.warning("No historical matches found")
            return pd.DataFrame()
        
        features = {}
        
        # Basic match info
        features['home_team_id'] = home_team_id
        features['away_team_id'] = away_team_id
        features['match_date'] = match_date
        
        # Form features for both teams
        for window in FEATURE_CONFIG['form_windows']:
            # Home team form
            home_form = self.form_features.calculate_last_n_form(
                matches_df, home_team_id, window, match_date
            )
            for key, value in home_form.items():
                features[f'home_{key}'] = value
            
            # Away team form
            away_form = self.form_features.calculate_last_n_form(
                matches_df, away_team_id, window, match_date
            )
            for key, value in away_form.items():
                features[f'away_{key}'] = value
        
        # Home/Away split form
        home_ha_form = self.form_features.calculate_home_away_form(
            matches_df, home_team_id, 5, match_date
        )
        for key, value in home_ha_form.items():
            features[f'home_{key}'] = value
        
        away_ha_form = self.form_features.calculate_home_away_form(
            matches_df, away_team_id, 5, match_date
        )
        for key, value in away_ha_form.items():
            features[f'away_{key}'] = value
        
        # Form trajectory
        features['home_form_trajectory'] = self.form_features.calculate_form_trajectory(
            matches_df, home_team_id, as_of_date=match_date
        )
        features['away_form_trajectory'] = self.form_features.calculate_form_trajectory(
            matches_df, away_team_id, as_of_date=match_date
        )
        
        # Weighted form
        features['home_weighted_form'] = self.form_features.calculate_weighted_form(
            matches_df, home_team_id, n=10, as_of_date=match_date
        )
        features['away_weighted_form'] = self.form_features.calculate_weighted_form(
            matches_df, away_team_id, n=10, as_of_date=match_date
        )
        
        # xG features
        for window in FEATURE_CONFIG['xg_windows']:
            # Home team xG
            home_xg = self.xg_features.calculate_xg_metrics(
                matches_df, home_team_id, window, match_date
            )
            for key, value in home_xg.items():
                features[f'home_{key}'] = value
            
            # Away team xG
            away_xg = self.xg_features.calculate_xg_metrics(
                matches_df, away_team_id, window, match_date
            )
            for key, value in away_xg.items():
                features[f'away_{key}'] = value
        
        # xG trends
        features['home_xg_trend'] = self.xg_features.calculate_xg_trend(
            matches_df, home_team_id, as_of_date=match_date
        )
        features['away_xg_trend'] = self.xg_features.calculate_xg_trend(
            matches_df, away_team_id, as_of_date=match_date
        )
        
        # Head-to-head features
        h2h_features = self._calculate_h2h_features(
            matches_df, home_team_id, away_team_id, match_date
        )
        features.update(h2h_features)
        
        # Differential features (home - away)
        features['ppg_differential'] = features.get('home_last_5_ppg', 0) - features.get('away_last_5_ppg', 0)
        features['xg_differential'] = features.get('home_last_5_avg_xg_for', 0) - features.get('away_last_5_avg_xg_for', 0)
        features['form_trajectory_differential'] = features['home_form_trajectory'] - features['away_form_trajectory']
        
        return pd.DataFrame([features])
    
    def create_training_dataset(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Create features for all historical matches
        
        Args:
            start_date: Start date for matches (optional)
            end_date: End date for matches (optional)
        """
        logger.info("Creating training dataset...")
        
        # Get all matches
        query = self.session.query(Match).filter(
            Match.home_goals.isnot(None),  # Only finished matches
            Match.away_goals.isnot(None)
        )
        
        if start_date:
            query = query.filter(Match.match_date >= start_date)
        if end_date:
            query = query.filter(Match.match_date <= end_date)
        
        query = query.order_by(Match.match_date)
        matches = query.all()
        
        logger.info(f"Processing {len(matches)} matches...")
        
        all_features = []
        
        for i, match in enumerate(matches):
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(matches)} matches")
            
            try:
                features = self.create_match_features(
                    match.home_team_id,
                    match.away_team_id,
                    match.match_date
                )
                
                if not features.empty:
                    # Add target variable
                    features['result'] = match.result
                    features['home_goals'] = match.home_goals
                    features['away_goals'] = match.away_goals
                    features['match_id'] = match.match_id
                    
                    all_features.append(features)
                    
            except Exception as e:
                logger.error(f"Error processing match {match.match_id}: {e}")
                continue
        
        if all_features:
            df = pd.concat(all_features, ignore_index=True)
            logger.info(f"Created training dataset with {len(df)} samples and {len(df.columns)} features")
            return df
        
        return pd.DataFrame()
    
    def _get_historical_matches(self, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical matches from database"""
        query = self.session.query(Match).filter(
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None)
        )
        
        if as_of_date:
            query = query.filter(Match.match_date < as_of_date)
        
        matches = query.all()
        
        data = []
        for match in matches:
            data.append({
                'match_id': match.match_id,
                'match_date': match.match_date,
                'home_team_id': match.home_team_id,
                'away_team_id': match.away_team_id,
                'home_goals': match.home_goals,
                'away_goals': match.away_goals,
                'home_xg': match.home_xg,
                'away_xg': match.away_xg,
                'result': match.result,
            })
        
        return pd.DataFrame(data)
    
    def _calculate_h2h_features(self, matches_df: pd.DataFrame, 
                                home_team_id: int, away_team_id: int,
                                as_of_date: datetime) -> dict:
        """Calculate head-to-head features"""
        # Get H2H matches
        h2h_matches = matches_df[
            (((matches_df['home_team_id'] == home_team_id) & (matches_df['away_team_id'] == away_team_id)) |
             ((matches_df['home_team_id'] == away_team_id) & (matches_df['away_team_id'] == home_team_id))) &
            (matches_df['match_date'] < as_of_date)
        ].sort_values('match_date', ascending=False).head(FEATURE_CONFIG['max_h2h_matches'])
        
        if len(h2h_matches) == 0:
            return {
                'h2h_total_matches': 0,
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_avg_home_goals': 0,
                'h2h_avg_away_goals': 0,
            }
        
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = []
        away_goals = []
        
        for _, match in h2h_matches.iterrows():
            if match['home_team_id'] == home_team_id:
                home_goals.append(match['home_goals'])
                away_goals.append(match['away_goals'])
                if match['result'] == 'H':
                    home_wins += 1
                elif match['result'] == 'A':
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals.append(match['away_goals'])
                away_goals.append(match['home_goals'])
                if match['result'] == 'A':
                    home_wins += 1
                elif match['result'] == 'H':
                    away_wins += 1
                else:
                    draws += 1
        
        return {
            'h2h_total_matches': len(h2h_matches),
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_avg_home_goals': np.mean(home_goals) if home_goals else 0,
            'h2h_avg_away_goals': np.mean(away_goals) if away_goals else 0,
            'h2h_home_win_rate': home_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0,
        }
    
    def __del__(self):
        """Close database session"""
        if hasattr(self, 'session'):
            self.session.close()


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    engineer = FeatureEngineer()
    
    # Create training dataset
    training_data = engineer.create_training_dataset()
    
    if not training_data.empty:
        print(f"\nTraining dataset shape: {training_data.shape}")
        print(f"Features: {training_data.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(training_data.head())
