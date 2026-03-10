"""
Multi-Stage Ensemble with Confidence Gating
Implements smart alignment logic for complementary predictions
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AlignmentEnsemble:
    """
    Ensemble predictor that aligns base, lineup, and score models
    with confidence calibration based on model agreement
    """
    
    def __init__(self):
        self.outcome_labels = {0: 'away_win', 1: 'draw', 2: 'home_win'}
        
        # Historical accuracy weights (update these based on performance)
        self.weights = {
            'base': 0.45,
            'lineup': 0.40,
            'score': 0.15
        }
        
        # Confidence multipliers based on agreement level
        self.confidence_boost = {
            'all_agree': 0.85,
            'two_agree': 0.70,
            'disagree': 0.55
        }
    
    def score_to_outcome(self, home_goals: int, away_goals: int) -> str:
        """Convert score prediction to outcome"""
        if home_goals > away_goals:
            return 'home_win'
        elif home_goals < away_goals:
            return 'away_win'
        else:
            return 'draw'
    
    def align_score_with_outcome(self, home_goals: int, away_goals: int, required_outcome: str) -> Tuple[int, int]:
        """
        Align score prediction with required outcome
        
        CRITICAL RULE: Score MUST match the outcome prediction
        - If outcome=Home Win, score CANNOT be draw or away win
        - If outcome=Draw, score CANNOT be home or away win
        - If outcome=Away Win, score CANNOT be draw or home win
        
        Args:
            home_goals: Predicted home goals
            away_goals: Predicted away goals
            required_outcome: The outcome that must be matched ('home_win', 'draw', 'away_win')
        
        Returns:
            Tuple of (adjusted_home_goals, adjusted_away_goals) that matches required_outcome
        """
        current_outcome = self.score_to_outcome(home_goals, away_goals)
        
        # If already aligned, return as is
        if current_outcome == required_outcome:
            return (home_goals, away_goals)
        
        # Need to adjust score to match required outcome
        total_goals = home_goals + away_goals
        
        if required_outcome == 'home_win':
            # Must have home_goals > away_goals
            if total_goals == 0:
                return (1, 0)  # Minimum home win
            elif home_goals == away_goals:
                # Draw → Home Win: add 1 to home
                return (home_goals + 1, away_goals)
            elif home_goals < away_goals:
                # Away Win → Home Win: flip and add 1 to home
                return (away_goals + 1, home_goals)
            else:
                return (home_goals, away_goals)
        
        elif required_outcome == 'away_win':
            # Must have away_goals > home_goals
            if total_goals == 0:
                return (0, 1)  # Minimum away win
            elif home_goals == away_goals:
                # Draw → Away Win: add 1 to away
                return (home_goals, away_goals + 1)
            elif away_goals < home_goals:
                # Home Win → Away Win: flip and add 1 to away
                return (away_goals, home_goals + 1)
            else:
                return (home_goals, away_goals)
        
        else:  # required_outcome == 'draw'
            # Must have home_goals == away_goals
            if home_goals > away_goals:
                # Home Win → Draw: reduce home to match away
                return (away_goals, away_goals)
            elif away_goals > home_goals:
                # Away Win → Draw: reduce away to match home
                return (home_goals, home_goals)
            else:
                return (home_goals, away_goals)
    
    def proba_to_outcome(self, proba: np.ndarray) -> str:
        """Convert probability array to outcome"""
        idx = np.argmax(proba)
        return self.outcome_labels[idx]
    
    def check_agreement(self, predictions: list) -> Dict:
        """
        Check how many models agree on the outcome
        
        Args:
            predictions: List of outcome strings ['home_win', 'draw', 'away_win']
        
        Returns:
            Dict with agreement level and details
        """
        unique_predictions = list(set(predictions))
        
        if len(unique_predictions) == 1:
            return {
                'level': 'all_agree',
                'agreement': 'full',
                'outcome': unique_predictions[0],
                'confidence_factor': self.confidence_boost['all_agree']
            }
        
        # Count votes for each outcome
        votes = {}
        for pred in predictions:
            votes[pred] = votes.get(pred, 0) + 1
        
        max_votes = max(votes.values())
        
        if max_votes >= 2:  # At least 2 models agree
            majority_outcome = [k for k, v in votes.items() if v == max_votes][0]
            return {
                'level': 'two_agree',
                'agreement': 'majority',
                'outcome': majority_outcome,
                'votes': votes,
                'confidence_factor': self.confidence_boost['two_agree']
            }
        
        # All disagree
        return {
            'level': 'disagree',
            'agreement': 'none',
            'outcome': None,
            'votes': votes,
            'confidence_factor': self.confidence_boost['disagree']
        }
    
    def weighted_average(self, probas: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Combine multiple probability predictions with weights
        
        Args:
            probas: Dict of {'model_name': proba_array}
            weights: Dict of {'model_name': weight}
        
        Returns:
            Combined probability array [away, draw, home]
        """
        result = np.zeros(3)
        total_weight = sum(weights.values())
        
        for model_name, proba in probas.items():
            weight = weights.get(model_name, 0)
            result += proba * weight
        
        return result / total_weight
    
    def _score_to_proba(self, outcome: str) -> np.ndarray:
        """
        Convert score-based outcome to probability distribution
        
        Args:
            outcome: 'home_win', 'draw', or 'away_win'
        
        Returns:
            Probability array [away, draw, home]
        """
        # Assign moderate confidence to score predictions
        # Score model is less certain about exact outcome
        if outcome == 'home_win':
            return np.array([0.15, 0.20, 0.65])
        elif outcome == 'away_win':
            return np.array([0.65, 0.20, 0.15])
        else:  # draw
            return np.array([0.20, 0.60, 0.20])
    
    def predict(
        self,
        base_proba: np.ndarray,
        score_pred: Tuple[int, int],
        lineup_proba: Optional[np.ndarray] = None,

        has_custom_formation: bool = False,
        score_model_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate aligned prediction with confidence calibration
        
        BHAKUNDO PRIORITY SYSTEM:
        1. If Base + Lineup agree → BHAKUNDO PREDICTS (highest confidence)
        2. If all 3 models disagree:
           - WITH custom formation: Lineup → Base → Score
           - WITHOUT custom formation: Base → Lineup → Score
        
        Args:
            base_proba: Base model probabilities [away, draw, home]
            score_pred: Tuple of (home_goals, away_goals)
            lineup_proba: Optional lineup model probabilities
            has_custom_formation: Whether user edited formations
        
        Returns:
            Dict with final prediction, confidence, agreement info
        """
        home_goals, away_goals = score_pred
        score_outcome = self.score_to_outcome(home_goals, away_goals)
        base_outcome = self.proba_to_outcome(base_proba)

        # Resolve score-model signal once: use calibrated proba if supplied,
        # otherwise fall back to the hard binary mapping.
        _score_proba = (np.array(score_model_proba, dtype=float)
                        if score_model_proba is not None
                        else self._score_to_proba(score_outcome))
        
        # Collect predictions
        predictions = [base_outcome, score_outcome]
        model_probas = {'base': base_proba}
        
        if lineup_proba is not None:
            lineup_outcome = self.proba_to_outcome(lineup_proba)
            predictions.append(lineup_outcome)
            model_probas['lineup'] = lineup_proba
        
        # Check agreement
        agreement = self.check_agreement(predictions)
        
        # BHAKUNDO PRIORITY LOGIC
        if agreement['level'] == 'all_agree':
            # All 3 models agree - use most informed (lineup if available)
            if lineup_proba is not None:
                final_proba = lineup_proba
                primary_model = 'lineup'
            else:
                final_proba = base_proba
                primary_model = 'base'
            confidence = agreement['confidence_factor'] * np.max(final_proba)
            prediction_source = 'All Models Agree'
            
        elif agreement['level'] == 'two_agree':
            # 2 models agree - BHAKUNDO SPECIAL CASE
            if lineup_proba is not None:
                # Check if base + lineup agree (BHAKUNDO PREDICTS!)
                if base_outcome == lineup_outcome:
                    # BHAKUNDO PREDICTION - Base + Lineup consensus
                    final_proba = (base_proba + lineup_proba) / 2
                    confidence = 0.88  # High confidence - BHAKUNDO verified
                    prediction_source = '🎯 BHAKUNDO PREDICTS (Base + Lineup Agree)'
                    primary_model = 'bhakundo'
                elif base_outcome == score_outcome:
                    # Base + Score agree (Lineup is minority)
                    final_proba = (base_proba + _score_proba) / 2
                    confidence = 0.75
                    prediction_source = 'Base + Score Agree'
                    primary_model = 'base'
                elif lineup_outcome == score_outcome:
                    # Lineup + Score agree (Base is minority)
                    final_proba = (lineup_proba + _score_proba) / 2
                    confidence = 0.72
                    prediction_source = 'Lineup + Score Agree'
                    primary_model = 'lineup'
                else:
                    # Shouldn't happen (two_agree means 2 must match)
                    weights = self.weights.copy()
                    final_proba = self.weighted_average(model_probas, weights)
                    confidence = agreement['confidence_factor'] * np.max(final_proba)
                    prediction_source = 'Weighted Consensus'
                    primary_model = 'weighted'
            else:
                # Only base and score
                if base_outcome == score_outcome:
                    final_proba = base_proba
                    confidence = 0.75
                    prediction_source = 'Base + Score Agree'
                    primary_model = 'base'
                else:
                    final_proba = base_proba * 0.7 + _score_proba * 0.3
                    confidence = 0.60
                    prediction_source = 'Base Priority'
                    primary_model = 'base'
            
        else:
            # All 3 models DISAGREE - Apply formation priority
            # PRIORITY RULES:
            # - With edited lineup: Lineup (highest) → Base → Score (never highest)
            # - Default lineup: Base (highest) → Lineup → Score (never highest)
            if lineup_proba is not None:
                if has_custom_formation:
                    # User edited formation → Lineup has highest priority
                    final_proba = lineup_proba * 0.55 + base_proba * 0.30 + _score_proba * 0.15
                    prediction_source = 'Lineup Priority (Edited Formation)'
                    primary_model = 'lineup'
                else:
                    # Default formation → Base has highest priority
                    final_proba = base_proba * 0.55 + lineup_proba * 0.30 + _score_proba * 0.15
                    prediction_source = 'Base Priority (Default Formation)'
                    primary_model = 'base'
            else:
                # Only base and score disagree - Base always wins
                final_proba = base_proba * 0.75 + _score_proba * 0.25
                prediction_source = 'Base Priority (No Lineup)'
                primary_model = 'base'
            
            confidence = agreement['confidence_factor'] * np.max(final_proba)
        
        # Convert final prediction
        final_outcome = self.proba_to_outcome(final_proba)
        
        # CRITICAL: Align score with final outcome
        # USER REQUIREMENT: Score MUST match outcome prediction
        # - If base says Home Win, score CANNOT be 2-2 (draw) or 1-2 (away win)
        # - If base says Draw, score CANNOT be 3-1 (home win) or 1-4 (away win)
        # - If base/lineup disagree, score follows base
        aligned_home, aligned_away = self.align_score_with_outcome(home_goals, away_goals, final_outcome)
        
        # Log if score was adjusted
        if aligned_home != home_goals or aligned_away != away_goals:
            logger.info(f"⚠️ Score aligned: {home_goals}-{away_goals} → {aligned_home}-{aligned_away} to match {final_outcome}")
        
        # Build result
        result = {
            'prediction': final_outcome,
            'probabilities': {
                'away_win': float(final_proba[0]),
                'draw': float(final_proba[1]),
                'home_win': float(final_proba[2])
            },
            'expected_score': {
                'home': aligned_home,  # Use aligned score
                'away': aligned_away   # Use aligned score
            },
            'confidence': float(min(0.95, confidence)),  # Cap at 95%
            'agreement': agreement,
            'model_used': primary_model,
            'prediction_source': prediction_source,
            'models_aligned': agreement['level'] in ['all_agree', 'two_agree'],
            'bhakundo_verified': primary_model == 'bhakundo'
        }
        
        return result
    
    def _score_to_proba(self, outcome: str) -> np.ndarray:
        """Convert outcome string to probability array"""
        proba = np.zeros(3)
        if outcome == 'away_win':
            proba[0] = 1.0
        elif outcome == 'draw':
            proba[1] = 1.0
        else:  # home_win
            proba[2] = 1.0
        return proba
    
    def calibrate_confidence(self, prediction: Dict, meta_features: Optional[Dict] = None) -> Dict:
        """
        Future enhancement: Use meta-model to calibrate confidence
        
        Args:
            prediction: Current prediction dict
            meta_features: Additional features for calibration
        
        Returns:
            Prediction with calibrated confidence
        """
        # TODO: Implement meta-model for confidence calibration
        # For now, just return as-is
        return prediction


def format_prediction_with_confidence(prediction: Dict) -> Dict:
    """
    Format prediction for frontend display with confidence indicators
    
    Args:
        prediction: Output from AlignmentEnsemble.predict()
    
    Returns:
        Formatted prediction for UI
    """
    confidence = prediction['confidence']
    
    # Confidence category
    if confidence >= 0.75:
        confidence_level = 'HIGH'
        confidence_color = 'green'
    elif confidence >= 0.60:
        confidence_level = 'MEDIUM'
        confidence_color = 'orange'
    else:
        confidence_level = 'LOW'
        confidence_color = 'red'
    
    agreement = prediction['agreement']
    
    return {
        **prediction,
        'confidence_display': {
            'value': confidence,
            'percentage': f"{confidence * 100:.1f}%",
            'level': confidence_level,
            'color': confidence_color,
            'text': f"{confidence_level} ({confidence * 100:.0f}%)"
        },
        'agreement_display': {
            'aligned': prediction['models_aligned'],
            'level': agreement['level'],
            'text': {
                'all_agree': '✓ All models agree',
                'two_agree': '~ Majority consensus',
                'disagree': '! Models disagree'
            }.get(agreement['level'], 'Unknown')
        }
    }
