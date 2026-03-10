"""xG (Expected Goals) features"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class XGFeatures:
    """Calculate Expected Goals related features"""
    
    def __init__(self):
        pass
    
    def calculate_xg_metrics(self, matches_df: pd.DataFrame, team_id: int,
                            n: int = 5, as_of_date: Optional[datetime] = None) -> dict:
        """
        Calculate xG metrics for last N games
        
        Returns:
            Dictionary with xG metrics
        """
        # Filter matches for this team
        team_matches = matches_df[
            ((matches_df['home_team_id'] == team_id) | 
             (matches_df['away_team_id'] == team_id)) &
            (matches_df['home_xg'].notna()) &  # Only matches with xG data
            (matches_df['away_xg'].notna())
        ].copy()
        
        # Filter by date if specified
        if as_of_date:
            team_matches = team_matches[team_matches['match_date'] < as_of_date]
        
        # Sort by date descending
        team_matches = team_matches.sort_values('match_date', ascending=False).head(n)
        
        if len(team_matches) == 0:
            return self._empty_xg_metrics(n)
        
        xg_for = []
        xg_against = []
        goals_for = []
        goals_against = []
        
        for _, match in team_matches.iterrows():
            if match['home_team_id'] == team_id:
                xg_for.append(match['home_xg'])
                xg_against.append(match['away_xg'])
                goals_for.append(match['home_goals'])
                goals_against.append(match['away_goals'])
            else:
                xg_for.append(match['away_xg'])
                xg_against.append(match['home_xg'])
                goals_for.append(match['away_goals'])
                goals_against.append(match['home_goals'])
        
        # Calculate metrics
        avg_xg_for = np.mean(xg_for)
        avg_xg_against = np.mean(xg_against)
        total_xg_for = sum(xg_for)
        total_xg_against = sum(xg_against)
        
        # xG overperformance
        xg_overperformance = sum(goals_for) - sum(xg_for)
        xg_defensive_overperformance = sum(xg_against) - sum(goals_against)
        
        return {
            f'last_{n}_avg_xg_for': avg_xg_for,
            f'last_{n}_avg_xg_against': avg_xg_against,
            f'last_{n}_total_xg_for': total_xg_for,
            f'last_{n}_total_xg_against': total_xg_against,
            f'last_{n}_xg_diff': total_xg_for - total_xg_against,
            f'last_{n}_xg_overperformance': xg_overperformance,
            f'last_{n}_xg_defensive_overperformance': xg_defensive_overperformance,
        }
    
    def calculate_xg_overperformance(self, matches_df: pd.DataFrame, team_id: int,
                                     n: int = 10, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate how much team over/underperforms xG
        Positive = scoring more than expected
        """
        metrics = self.calculate_xg_metrics(matches_df, team_id, n, as_of_date)
        return metrics.get(f'last_{n}_xg_overperformance', 0)
    
    def calculate_xg_trend(self, matches_df: pd.DataFrame, team_id: int,
                          n_recent: int = 5, n_previous: int = 5,
                          as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate xG trend (improving/declining attacking threat)
        """
        # Recent xG
        recent_metrics = self.calculate_xg_metrics(matches_df, team_id, n_recent, as_of_date)
        recent_avg_xg = recent_metrics[f'last_{n_recent}_avg_xg_for']
        
        # Previous xG
        team_matches = matches_df[
            ((matches_df['home_team_id'] == team_id) | 
             (matches_df['away_team_id'] == team_id)) &
            (matches_df['home_xg'].notna()) &
            (matches_df['away_xg'].notna())
        ].copy()
        
        if as_of_date:
            team_matches = team_matches[team_matches['match_date'] < as_of_date]
        
        team_matches = team_matches.sort_values('match_date', ascending=False)
        previous_matches = team_matches.iloc[n_recent:n_recent + n_previous]
        
        if len(previous_matches) == 0:
            return 0.0
        
        previous_xg = []
        for _, match in previous_matches.iterrows():
            if match['home_team_id'] == team_id:
                previous_xg.append(match['home_xg'])
            else:
                previous_xg.append(match['away_xg'])
        
        previous_avg_xg = np.mean(previous_xg) if previous_xg else 0
        
        return recent_avg_xg - previous_avg_xg
    
    def _empty_xg_metrics(self, n: int) -> dict:
        """Return empty metrics when no xG data available"""
        return {
            f'last_{n}_avg_xg_for': 0,
            f'last_{n}_avg_xg_against': 0,
            f'last_{n}_total_xg_for': 0,
            f'last_{n}_total_xg_against': 0,
            f'last_{n}_xg_diff': 0,
            f'last_{n}_xg_overperformance': 0,
            f'last_{n}_xg_defensive_overperformance': 0,
        }


if __name__ == "__main__":
    # Test xG features
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with xG
    sample_data = pd.DataFrame({
        'match_id': range(1, 11),
        'match_date': pd.date_range('2024-01-01', periods=10, freq='W'),
        'home_team_id': [1] * 5 + [2] * 5,
        'away_team_id': [2] * 5 + [1] * 5,
        'home_goals': [2, 1, 3, 0, 2, 1, 2, 1, 2, 3],
        'away_goals': [1, 1, 2, 1, 0, 2, 1, 1, 2, 1],
        'home_xg': [1.8, 1.5, 2.5, 0.8, 2.1, 1.2, 1.9, 1.3, 2.0, 2.8],
        'away_xg': [1.2, 0.9, 1.8, 1.5, 0.5, 1.8, 1.1, 1.0, 1.9, 1.2],
    })
    
    xg_features = XGFeatures()
    
    # Test xG metrics
    xg_metrics = xg_features.calculate_xg_metrics(sample_data, team_id=1, n=5)
    print("xG metrics:", xg_metrics)
    
    # Test xG trend
    xg_trend = xg_features.calculate_xg_trend(sample_data, team_id=1)
    print(f"\nxG trend: {xg_trend:.2f}")
