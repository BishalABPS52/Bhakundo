"""Score Prediction Model - Predicts exact scorelines"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class ScorePredictionModel:
    """
    Score prediction using Poisson distribution based on xG
    Predicts most likely scorelines and probabilities
    """
    
    def __init__(self):
        self.max_goals = 7  # Maximum goals to consider for each team
        self.rho = -0.13   # Dixon-Coles correction calibrated for Premier League
    
    def _dixon_coles_tau(self, home_goals: int, away_goals: int,
                          home_xg: float, away_xg: float) -> float:
        """
        Dixon-Coles correction for low-scoring scorelines.
        Fixes standard Poisson over-predicting 0-0 and under-predicting 1-0, 0-1, 1-1.
        Only affects scores where both teams score 0 or 1 goal.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        else:
            return 1.0

    def predict_scoreline_probabilities(self, home_xg: float, away_xg: float) -> List[Tuple[str, float]]:
        """
        Predict scoreline probabilities using Poisson distribution with
        Dixon-Coles correction for low-scoring scorelines.

        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team

        Returns:
            List of (scoreline, probability) tuples, sorted by probability descending.
            Probabilities are re-normalised to sum to 1.0 after DC correction.
        """
        scoreline_probs = []

        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                prob_home = poisson.pmf(home_goals, home_xg)
                prob_away = poisson.pmf(away_goals, away_xg)
                tau = self._dixon_coles_tau(home_goals, away_goals, home_xg, away_xg)
                prob = prob_home * prob_away * tau
                scoreline_probs.append((f"{home_goals}-{away_goals}", prob))

        # Re-normalise so probabilities sum to 1.0 after DC adjustment
        total = sum(p for _, p in scoreline_probs)
        scoreline_probs = [(s, p / total) for s, p in scoreline_probs]

        scoreline_probs.sort(key=lambda x: x[1], reverse=True)
        return scoreline_probs
    
    def constrain_score_to_outcome(self, home_goals: int, away_goals: int,
                                     predicted_outcome: str) -> tuple:
        """
        Ensures the predicted scoreline is directionally consistent with the
        predicted outcome.  Example: outcome='home_win' but score is 0-1 → fix to 1-0.

        Args:
            home_goals: Predicted home goals (int)
            away_goals: Predicted away goals (int)
            predicted_outcome: 'home_win', 'draw', or 'away_win'

        Returns:
            (home_goals, away_goals) corrected and clamped to [0, 5]
        """
        if predicted_outcome == 'home_win':
            if home_goals <= away_goals:
                home_goals = away_goals + 1
        elif predicted_outcome == 'away_win':
            if away_goals <= home_goals:
                away_goals = home_goals + 1
        elif predicted_outcome == 'draw':
            if home_goals != away_goals:
                equal_goals = min(home_goals, away_goals)
                home_goals = equal_goals
                away_goals = equal_goals

        return int(np.clip(home_goals, 0, 5)), int(np.clip(away_goals, 0, 5))

    def predict_most_likely_score(self, home_xg: float, away_xg: float,
                                   predicted_outcome: str = None) -> Tuple[int, int, float]:
        """
        Predict most likely scoreline, optionally constrained to match a predicted outcome.

        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            predicted_outcome: Optional — 'home_win', 'draw', or 'away_win'.
                               When provided, the returned score is guaranteed to
                               be directionally consistent with this outcome.

        Returns:
            (home_goals, away_goals, probability)
        """
        scorelines = self.predict_scoreline_probabilities(home_xg, away_xg)
        scoreline_str, probability = scorelines[0]
        home_goals, away_goals = map(int, scoreline_str.split('-'))

        if predicted_outcome is not None:
            home_goals, away_goals = self.constrain_score_to_outcome(
                home_goals, away_goals, predicted_outcome
            )

        return home_goals, away_goals, probability
    
    def predict_top_n_scores(self, home_xg: float, away_xg: float, n: int = 5) -> pd.DataFrame:
        """
        Get top N most likely scorelines
        
        Returns:
            DataFrame with scorelines and probabilities
        """
        scorelines = self.predict_scoreline_probabilities(home_xg, away_xg)
        top_n = scorelines[:n]
        
        df = pd.DataFrame(top_n, columns=['scoreline', 'probability'])
        df['probability_pct'] = df['probability'] * 100
        
        return df
    
    def predict_outcome_from_score_probs(self, home_xg: float, away_xg: float) -> dict:
        """
        Calculate outcome probabilities (H/D/A) from score predictions
        
        Returns:
            Dictionary with outcome probabilities
        """
        scorelines = self.predict_scoreline_probabilities(home_xg, away_xg)
        
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
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
    
    def predict_over_under(self, home_xg: float, away_xg: float, threshold: float = 2.5) -> dict:
        """
        Predict over/under probabilities for total goals
        
        Args:
            threshold: Goals threshold (e.g., 2.5 for over/under 2.5)
        """
        scorelines = self.predict_scoreline_probabilities(home_xg, away_xg)
        
        over_prob = 0
        under_prob = 0
        
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
    
    def predict_both_teams_to_score(self, home_xg: float, away_xg: float) -> dict:
        """
        Predict probability of both teams scoring (BTTS)
        """
        scorelines = self.predict_scoreline_probabilities(home_xg, away_xg)
        
        btts_yes = 0
        btts_no = 0
        
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

    def score_proba_to_outcome_probs(self, home_xg: float, away_xg: float) -> dict:
        """
        Complete probability output from xG values using Poisson + Dixon-Coles.
        Returns outcome probabilities, top-5 scores, over/under 2.5, and BTTS.

        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team

        Returns:
            dict with score_model_home_prob, score_model_draw_prob, score_model_away_prob,
            predicted_home_goals, predicted_away_goals, top_score_probability,
            top_5_scores, over_2_5_prob, btts_prob
        """
        scorelines = self.predict_scoreline_probabilities(home_xg, away_xg)

        home_prob = draw_prob = away_prob = 0.0
        over25_prob = btts_prob = 0.0

        for scoreline, prob in scorelines:
            hg, ag = map(int, scoreline.split('-'))
            if hg > ag:
                home_prob += prob
            elif hg < ag:
                away_prob += prob
            else:
                draw_prob += prob
            if hg + ag > 2:
                over25_prob += prob
            if hg > 0 and ag > 0:
                btts_prob += prob

        best_score = scorelines[0][0]
        top5 = [{'score': s, 'probability': round(p * 100, 1)} for s, p in scorelines[:5]]

        return {
            'score_model_home_prob':  float(home_prob),
            'score_model_draw_prob':  float(draw_prob),
            'score_model_away_prob':  float(away_prob),
            'predicted_home_goals':   int(best_score.split('-')[0]),
            'predicted_away_goals':   int(best_score.split('-')[1]),
            'top_score_probability':  float(scorelines[0][1]),
            'top_5_scores':           top5,
            'over_2_5_prob':          float(over25_prob),
            'btts_prob':              float(btts_prob),
        }


if __name__ == "__main__":
    # Test score prediction
    logging.basicConfig(level=logging.INFO)
    
    model = ScorePredictionModel()
    
    # Test with realistic xG values
    home_xg = 1.8
    away_xg = 1.2
    
    print(f"\nPredicting scores for home xG: {home_xg}, away xG: {away_xg}")
    
    # Most likely score
    home_goals, away_goals, prob = model.predict_most_likely_score(home_xg, away_xg)
    print(f"\nMost likely score: {home_goals}-{away_goals} (probability: {prob:.2%})")
    
    # Top 5 scorelines
    top_scores = model.predict_top_n_scores(home_xg, away_xg, n=5)
    print(f"\nTop 5 most likely scorelines:")
    print(top_scores)
    
    # Outcome probabilities
    outcomes = model.predict_outcome_from_score_probs(home_xg, away_xg)
    print(f"\nOutcome probabilities:")
    for outcome, prob in outcomes.items():
        print(f"  {outcome}: {prob:.2%}")
    
    # Over/Under
    over_under = model.predict_over_under(home_xg, away_xg, 2.5)
    print(f"\nOver/Under 2.5 goals:")
    for key, prob in over_under.items():
        print(f"  {key}: {prob:.2%}")
    
    # Both teams to score
    btts = model.predict_both_teams_to_score(home_xg, away_xg)
    print(f"\nBoth Teams To Score:")
    for key, prob in btts.items():
        print(f"  {key}: {prob:.2%}")
