"""Compare predictions from base model vs lineup-based model"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import ManualEnsemble so it can be unpickled
from scripts.train_final_model import ManualEnsemble
from src.models.lineup_models import LineupResultModel

import joblib
import pandas as pd
import numpy as np

def load_models():
    """Load both models"""
    base_model = joblib.load('data/models/outcome_model_final.pkl')
    lineup_model = joblib.load('data/models/lineup_result_model.pkl')
    return base_model, lineup_model

def compare_predictions():
    """Compare predictions on test set"""
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: Base vs Lineup-Based")
    print(f"{'='*80}\n")
    
    # Load models
    base_model, lineup_model = load_models()
    
    print(f"Base Model (86 features):")
    print(f"  Training Date: {base_model['training_date']}")
    print(f"  Test Accuracy: {base_model['test_accuracy']:.2%}")
    print(f"  Features: {base_model['n_features']}")
    
    print(f"\nLineup Model (97 features):")
    print(f"  Training Date: {lineup_model['training_date']}")
    print(f"  Test Accuracy: {lineup_model['test_accuracy']:.2%}")
    print(f"  Features: {lineup_model['n_features']}")
    print(f"  Additional: {lineup_model['n_features'] - base_model['n_features']} lineup features")
    
    print(f"\n{'='*80}")
    print(f"KEY DIFFERENCES")
    print(f"{'='*80}")
    print(f"\n1. Feature Set:")
    print(f"   Base: Form, momentum, venue stats, H2H, rest days")
    print(f"   Lineup: All above + formations, tactical matchups, aggression scores")
    
    print(f"\n2. Tactical Analysis:")
    print(f"   Base: ❌ Not available")
    print(f"   Lineup: ✅ Formation breakdown, midfield battle, def vs att")
    
    print(f"\n3. Formation Predictions:")
    print(f"   Base: ❌ Not included")
    print(f"   Lineup: ✅ Predicted formations for both teams")
    
    print(f"\n4. Use Cases:")
    print(f"   Base: Quick predictions, historical analysis")
    print(f"   Lineup: Tactical insights, formation planning, detailed analysis")
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    print(f"\nUse LINEUP MODEL when:")
    print(f"  • You need tactical insights and formation analysis")
    print(f"  • Predicting high-profile matches (derbies, top 6)")
    print(f"  • You have or can predict team formations")
    print(f"  • You want to understand WHY a prediction is made")
    
    print(f"\nUse BASE MODEL when:")
    print(f"  • You need quick predictions without tactical detail")
    print(f"  • Processing large batches of matches")
    print(f"  • Formations are unknown/unavailable")
    print(f"  • You only care about win/draw/loss probabilities")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    compare_predictions()
