"""
Poisson Score Predictor - Uses expected goals (xG) to predict scorelines
More accurate than direct score regression models
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class PoissonScorePredictor:
    """
    Predicts match scores using Poisson distribution based on expected goals (xG)
    This is more statistically sound than direct regression
    """
    
    def __init__(self, max_goals: int = 7):
        """
        Args:
            max_goals: Maximum goals to consider for each team
        """
        self.max_goals = max_goals
    
    def calculate_scoreline_probabilities(
        self, 
        home_xg: float, 
        away_xg: float
    ) -> List[Tuple[str, float]]:
        """
        Calculate probability for all possible scorelines using Poisson distribution
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            
        Returns:
            List of (scoreline, probability) tuples, sorted by probability (highest first)
        """
        scoreline_probs = []
        
        # Apply home advantage adjustment (typically 0.3-0.4 goals)
        home_xg_adjusted = home_xg * 1.15  # Home teams score ~15% more
        away_xg_adjusted = away_xg * 0.95  # Away teams score ~5% less
        
        # Calculate probability for each possible scoreline
        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                # Probability using Poisson distribution
                prob_home = poisson.pmf(home_goals, home_xg_adjusted)
                prob_away = poisson.pmf(away_goals, away_xg_adjusted)
                
                # Combined probability (independent events)
                combined_prob = prob_home * prob_away
                
                scoreline = f"{home_goals}-{away_goals}"
                scoreline_probs.append((scoreline, combined_prob))
        
        # Sort by probability (descending)
        scoreline_probs.sort(key=lambda x: x[1], reverse=True)
        
        return scoreline_probs
    
    def predict_most_likely_score(
        self, 
        home_xg: float, 
        away_xg: float
    ) -> Tuple[int, int, float]:
        """
        Predict the single most likely scoreline
        
        Returns:
            Tuple of (home_goals, away_goals, probability)
        """
        scorelines = self.calculate_scoreline_probabilities(home_xg, away_xg)
        
        # Get the most likely scoreline
        best_scoreline, best_prob = scorelines[0]
        home_goals, away_goals = map(int, best_scoreline.split('-'))
        
        logger.info(f"Most likely score: {best_scoreline} (probability: {best_prob:.2%})")
        
        return home_goals, away_goals, best_prob
    
    def get_top_n_scorelines(
        self, 
        home_xg: float, 
        away_xg: float, 
        n: int = 5
    ) -> pd.DataFrame:
        """
        Get the top N most likely scorelines
        
        Returns:
            DataFrame with columns: scoreline, probability, probability_pct
        """
        scorelines = self.calculate_scoreline_probabilities(home_xg, away_xg)
        top_n = scorelines[:n]
        
        df = pd.DataFrame(top_n, columns=['scoreline', 'probability'])
        df['probability_pct'] = df['probability'] * 100
        
        return df
    
    def calculate_outcome_probabilities(
        self, 
        home_xg: float, 
        away_xg: float
    ) -> Dict[str, float]:
        """
        Calculate home/draw/away probabilities from score predictions
        
        Returns:
            Dict with keys: home_win_prob, draw_prob, away_win_prob
        """
        scorelines = self.calculate_scoreline_probabilities(home_xg, away_xg)
        
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        for scoreline, prob in scorelines:
            home_goals, away_goals = map(int, scoreline.split('-'))
            
            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals < away_goals:
                away_win_prob += prob
            else:
                draw_prob += prob
        
        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob
        }
    
    def predict_over_under(
        self, 
        home_xg: float, 
        away_xg: float, 
        threshold: float = 2.5
    ) -> Dict[str, float]:
        """
        Calculate probability of total goals being over/under threshold
        
        Args:
            threshold: Goals threshold (e.g., 2.5)
            
        Returns:
            Dict with over and under probabilities
        """
        scorelines = self.calculate_scoreline_probabilities(home_xg, away_xg)
        
        over_prob = 0.0
        under_prob = 0.0
        
        for scoreline, prob in scorelines:
            home_goals, away_goals = map(int, scoreline.split('-'))
            total_goals = home_goals + away_goals
            
            if total_goals > threshold:
                over_prob += prob
            else:
                under_prob += prob
        
        return {
            f'over_{threshold}': over_prob,
            f'under_{threshold}': under_prob
        }
    
    def predict_btts(
        self, 
        home_xg: float, 
        away_xg: float
    ) -> Dict[str, float]:
        """
        Predict Both Teams To Score (BTTS) probability
        
        Returns:
            Dict with btts_yes and btts_no probabilities
        """
        scorelines = self.calculate_scoreline_probabilities(home_xg, away_xg)
        
        btts_yes = 0.0
        btts_no = 0.0
        
        for scoreline, prob in scorelines:
            home_goals, away_goals = map(int, scoreline.split('-'))
            
            if home_goals > 0 and away_goals > 0:
                btts_yes += prob
            else:
                btts_no += prob
        
        return {
            'btts_yes': btts_yes,
            'btts_no': btts_no
        }
    
    def extract_xg_from_features(self, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Extract expected goals from feature DataFrame
        
        Priority order:
        1. home_avg_xg_for / away_avg_xg_for
        2. home_last_5_avg_xg_for / away_last_5_avg_xg_for  
        3. home_avg_goals_scored / away_avg_goals_scored
        4. Default to 1.5 and 1.2
        
        Returns:
            Tuple of (home_xg, away_xg)
        """
        # Try different feature names
        xg_columns = [
            ('home_avg_xg_for', 'away_avg_xg_for'),
            ('home_last_5_avg_xg_for', 'away_last_5_avg_xg_for'),
            ('home_xg_avg', 'away_xg_avg'),
            ('home_avg_goals_scored', 'away_avg_goals_scored'),
        ]
        
        for home_col, away_col in xg_columns:
            if home_col in features.columns and away_col in features.columns:
                home_xg = features[home_col].iloc[0]
                away_xg = features[away_col].iloc[0]
                
                # Ensure reasonable values
                home_xg = max(0.1, min(5.0, home_xg))
                away_xg = max(0.1, min(5.0, away_xg))
                
                logger.info(f"Using xG from {home_col}: home={home_xg:.2f}, away={away_xg:.2f}")
                return home_xg, away_xg
        
        # Fallback to defaults
        logger.warning("No xG features found, using defaults: home=1.5, away=1.2")
        return 1.5, 1.2


if __name__ == "__main__":
    # Test the Poisson predictor
    logging.basicConfig(level=logging.INFO)
    
    predictor = PoissonScorePredictor()
    
    # Test case: Strong home team vs average away team
    home_xg = 2.1  # Strong home attack
    away_xg = 1.3  # Average away attack
    
    print(f"\n{'='*60}")
    print(f"TESTING POISSON SCORE PREDICTOR")
    print(f"{'='*60}")
    print(f"Home xG: {home_xg} | Away xG: {away_xg}")
    
    # Most likely score
    h_goals, a_goals, prob = predictor.predict_most_likely_score(home_xg, away_xg)
    print(f"\n✅ Most Likely Score: {h_goals}-{a_goals} (Probability: {prob:.2%})")
    
    # Top 5 scorelines
    top_5 = predictor.get_top_n_scorelines(home_xg, away_xg, n=5)
    print(f"\n📊 Top 5 Most Likely Scorelines:")
    print(top_5.to_string(index=False))
    
    # Outcome probabilities
    outcomes = predictor.calculate_outcome_probabilities(home_xg, away_xg)
    print(f"\n🎯 Outcome Probabilities:")
    print(f"   Home Win: {outcomes['home_win_prob']:.2%}")
    print(f"   Draw:     {outcomes['draw_prob']:.2%}")
    print(f"   Away Win: {outcomes['away_win_prob']:.2%}")
    
    # Over/Under 2.5
    over_under = predictor.predict_over_under(home_xg, away_xg, 2.5)
    print(f"\n📈 Over/Under 2.5 Goals:")
    print(f"   Over:  {over_under['over_2.5']:.2%}")
    print(f"   Under: {over_under['under_2.5']:.2%}")
    
    # Both teams to score
    btts = predictor.predict_btts(home_xg, away_xg)
    print(f"\n⚽ Both Teams To Score:")
    print(f"   Yes: {btts['btts_yes']:.2%}")
    print(f"   No:  {btts['btts_no']:.2%}")
    
    print(f"\n{'='*60}\n")
