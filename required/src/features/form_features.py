"""Form features - Team form calculations and metrics"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FormFeatures:
    """Calculate team form metrics"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_points(result: str) -> int:
        """Convert result to points (W=3, D=1, L=0)"""
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0
    
    def calculate_last_n_form(self, matches_df: pd.DataFrame, team_id: int, 
                              n: int = 5, as_of_date: Optional[datetime] = None) -> dict:
        """
        Calculate form metrics for last N games
        
        Args:
            matches_df: DataFrame with match data
            team_id: Team ID
            n: Number of games to consider
            as_of_date: Calculate form as of this date
            
        Returns:
            Dictionary with form metrics
        """
        # Filter matches for this team
        team_matches = matches_df[
            ((matches_df['home_team_id'] == team_id) | 
             (matches_df['away_team_id'] == team_id))
        ].copy()
        
        # Filter by date if specified
        if as_of_date:
            team_matches = team_matches[team_matches['match_date'] < as_of_date]
        
        # Sort by date descending
        team_matches = team_matches.sort_values('match_date', ascending=False)
        
        # Get last N matches
        last_n_matches = team_matches.head(n)
        
        if len(last_n_matches) == 0:
            return self._empty_form_metrics(n)
        
        # Calculate results
        results = []
        goals_for = []
        goals_against = []
        
        for _, match in last_n_matches.iterrows():
            if match['home_team_id'] == team_id:
                gf = match['home_goals']
                ga = match['away_goals']
            else:
                gf = match['away_goals']
                ga = match['home_goals']
            
            goals_for.append(gf)
            goals_against.append(ga)
            
            if gf > ga:
                results.append('W')
            elif gf < ga:
                results.append('L')
            else:
                results.append('D')
        
        # Calculate metrics
        form_string = ''.join(results)
        points = sum([self.calculate_points(r) for r in results])
        wins = results.count('W')
        draws = results.count('D')
        losses = results.count('L')
        
        return {
            f'last_{n}_form': form_string,
            f'last_{n}_points': points,
            f'last_{n}_ppg': points / len(results) if results else 0,
            f'last_{n}_wins': wins,
            f'last_{n}_draws': draws,
            f'last_{n}_losses': losses,
            f'last_{n}_goals_for': sum(goals_for),
            f'last_{n}_goals_against': sum(goals_against),
            f'last_{n}_goal_diff': sum(goals_for) - sum(goals_against),
            f'last_{n}_avg_goals_for': np.mean(goals_for) if goals_for else 0,
            f'last_{n}_avg_goals_against': np.mean(goals_against) if goals_against else 0,
            f'last_{n}_clean_sheets': sum([1 for ga in goals_against if ga == 0]),
        }
    
    def calculate_home_away_form(self, matches_df: pd.DataFrame, team_id: int,
                                  n: int = 5, as_of_date: Optional[datetime] = None) -> dict:
        """Calculate separate home and away form"""
        # Home form
        home_matches = matches_df[matches_df['home_team_id'] == team_id].copy()
        if as_of_date:
            home_matches = home_matches[home_matches['match_date'] < as_of_date]
        
        home_form = self._calculate_form_from_matches(home_matches, team_id, n, is_home=True)
        
        # Away form
        away_matches = matches_df[matches_df['away_team_id'] == team_id].copy()
        if as_of_date:
            away_matches = away_matches[away_matches['match_date'] < as_of_date]
        
        away_form = self._calculate_form_from_matches(away_matches, team_id, n, is_home=False)
        
        # Combine with prefixes
        result = {}
        for key, value in home_form.items():
            result[f'home_{key}'] = value
        for key, value in away_form.items():
            result[f'away_{key}'] = value
        
        return result
    
    def _calculate_form_from_matches(self, matches_df: pd.DataFrame, team_id: int,
                                     n: int, is_home: bool) -> dict:
        """Helper function to calculate form from filtered matches"""
        matches_df = matches_df.sort_values('match_date', ascending=False).head(n)
        
        if len(matches_df) == 0:
            return self._empty_form_metrics(n, prefix='')
        
        results = []
        goals_for = []
        goals_against = []
        
        for _, match in matches_df.iterrows():
            if is_home:
                gf, ga = match['home_goals'], match['away_goals']
            else:
                gf, ga = match['away_goals'], match['home_goals']
            
            goals_for.append(gf)
            goals_against.append(ga)
            
            if gf > ga:
                results.append('W')
            elif gf < ga:
                results.append('L')
            else:
                results.append('D')
        
        points = sum([self.calculate_points(r) for r in results])
        
        return {
            'form_string': ''.join(results),
            'points': points,
            'ppg': points / len(results) if results else 0,
            'wins': results.count('W'),
            'goals_for': sum(goals_for),
            'goals_against': sum(goals_against),
        }
    
    def calculate_form_trajectory(self, matches_df: pd.DataFrame, team_id: int,
                                   n_recent: int = 5, n_previous: int = 5,
                                   as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate form trajectory (improving/declining)
        Returns positive value if improving, negative if declining
        """
        team_matches = matches_df[
            ((matches_df['home_team_id'] == team_id) | 
             (matches_df['away_team_id'] == team_id))
        ].copy()
        
        if as_of_date:
            team_matches = team_matches[team_matches['match_date'] < as_of_date]
        
        team_matches = team_matches.sort_values('match_date', ascending=False)
        
        if len(team_matches) < (n_recent + n_previous):
            return 0.0
        
        # Recent form
        recent_form = self.calculate_last_n_form(matches_df, team_id, n_recent, as_of_date)
        recent_ppg = recent_form[f'last_{n_recent}_ppg']
        
        # Previous form (skip recent matches)
        previous_matches = team_matches.iloc[n_recent:n_recent + n_previous]
        previous_results = []
        
        for _, match in previous_matches.iterrows():
            if match['home_team_id'] == team_id:
                gf, ga = match['home_goals'], match['away_goals']
            else:
                gf, ga = match['away_goals'], match['home_goals']
            
            if gf > ga:
                previous_results.append('W')
            elif gf < ga:
                previous_results.append('L')
            else:
                previous_results.append('D')
        
        previous_points = sum([self.calculate_points(r) for r in previous_results])
        previous_ppg = previous_points / len(previous_results) if previous_results else 0
        
        # Trajectory is difference
        return recent_ppg - previous_ppg
    
    def calculate_weighted_form(self, matches_df: pd.DataFrame, team_id: int,
                                n: int = 10, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate weighted form (recent matches weighted more heavily)
        Uses exponential decay: weight = e^(-lambda * i) where i is match index
        """
        team_matches = matches_df[
            ((matches_df['home_team_id'] == team_id) | 
             (matches_df['away_team_id'] == team_id))
        ].copy()
        
        if as_of_date:
            team_matches = team_matches[team_matches['match_date'] < as_of_date]
        
        team_matches = team_matches.sort_values('match_date', ascending=False).head(n)
        
        if len(team_matches) == 0:
            return 0.0
        
        lambda_decay = 0.15
        weighted_points = 0
        total_weight = 0
        
        for i, (_, match) in enumerate(team_matches.iterrows()):
            weight = np.exp(-lambda_decay * i)
            
            if match['home_team_id'] == team_id:
                gf, ga = match['home_goals'], match['away_goals']
            else:
                gf, ga = match['away_goals'], match['home_goals']
            
            if gf > ga:
                points = 3
            elif gf < ga:
                points = 0
            else:
                points = 1
            
            weighted_points += points * weight
            total_weight += weight
        
        return weighted_points / total_weight if total_weight > 0 else 0
    
    def _empty_form_metrics(self, n: int, prefix: str = '') -> dict:
        """Return empty metrics when no matches available"""
        if prefix:
            prefix = f'{prefix}_'
        
        return {
            f'{prefix}last_{n}_form': '',
            f'{prefix}last_{n}_points': 0,
            f'{prefix}last_{n}_ppg': 0,
            f'{prefix}last_{n}_wins': 0,
            f'{prefix}last_{n}_draws': 0,
            f'{prefix}last_{n}_losses': 0,
            f'{prefix}last_{n}_goals_for': 0,
            f'{prefix}last_{n}_goals_against': 0,
            f'{prefix}last_{n}_goal_diff': 0,
            f'{prefix}last_{n}_avg_goals_for': 0,
            f'{prefix}last_{n}_avg_goals_against': 0,
            f'{prefix}last_{n}_clean_sheets': 0,
        }


if __name__ == "__main__":
    # Test form features
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'match_id': range(1, 11),
        'match_date': pd.date_range('2024-01-01', periods=10, freq='W'),
        'home_team_id': [1] * 5 + [2] * 5,
        'away_team_id': [2] * 5 + [1] * 5,
        'home_goals': [2, 1, 3, 0, 2, 1, 2, 1, 2, 3],
        'away_goals': [1, 1, 2, 1, 0, 2, 1, 1, 2, 1],
    })
    
    form_features = FormFeatures()
    
    # Test last N form
    form = form_features.calculate_last_n_form(sample_data, team_id=1, n=5)
    print("Last 5 form:", form)
    
    # Test home/away form
    home_away = form_features.calculate_home_away_form(sample_data, team_id=1, n=3)
    print("\nHome/Away form:", home_away)
    
    # Test form trajectory
    trajectory = form_features.calculate_form_trajectory(sample_data, team_id=1)
    print(f"\nForm trajectory: {trajectory:.2f}")
