"""Build training dataset with engineered features for model training"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_training_dataset():
    """Build comprehensive training dataset with all engineered features"""
    
    logger.info("="*60)
    logger.info("Building Training Dataset")
    logger.info("="*60)
    
    # 1. Load match data
    logger.info("\n1. Loading match data...")
    matches_df = pd.read_csv('data/raw/matches_combined.csv')
    logger.info(f"   Loaded {len(matches_df)} matches")
    
    # 2. Convert date to datetime
    matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
    matches_df = matches_df.sort_values('match_date').reset_index(drop=True)
    
    # 3. Initialize feature engineer
    logger.info("\n2. Engineering features...")
    feature_engineer = FeatureEngineer()
    
    # 4. Create features for each match
    features_list = []
    
    for idx, match in matches_df.iterrows():
        if idx % 100 == 0:
            logger.info(f"   Processing match {idx+1}/{len(matches_df)}")
        
        try:
            features = feature_engineer.create_match_features(
                home_team=match['home_team'],
                away_team=match['away_team'],
                match_date=match['match_date'],
                matches_df=matches_df[:idx] if idx > 0 else pd.DataFrame()  # Only use past matches
            )
            
            if features is not None:
                # Add target variables
                features['home_goals'] = match['home_goals']
                features['away_goals'] = match['away_goals']
                features['goal_diff'] = match['home_goals'] - match['away_goals']
                
                # Create outcome target (H=0, D=1, A=2)
                if match['home_goals'] > match['away_goals']:
                    features['outcome'] = 'H'  # Home win
                elif match['home_goals'] < match['away_goals']:
                    features['outcome'] = 'A'  # Away win
                else:
                    features['outcome'] = 'D'  # Draw
                
                features['match_date'] = match['match_date']
                features['gameweek'] = match.get('gameweek', 0)
                features['season'] = match.get('season', '')
                
                features_list.append(features)
        except Exception as e:
            logger.warning(f"   Error processing match {idx}: {e}")
            continue
    
    # 5. Create DataFrame
    logger.info(f"\n3. Creating feature dataset...")
    features_df = pd.DataFrame(features_list)
    
    # Remove rows with missing features (early matches without enough history)
    initial_len = len(features_df)
    features_df = features_df.dropna(subset=['home_form_last5', 'away_form_last5'])
    logger.info(f"   Removed {initial_len - len(features_df)} matches without sufficient history")
    logger.info(f"   Final dataset: {len(features_df)} matches with {features_df.shape[1]} features")
    
    # 6. Save dataset
    output_path = 'data/processed/training_dataset.csv'
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved training dataset to {output_path}")
    
    # 7. Display summary
    logger.info("\n" + "="*60)
    logger.info("Dataset Summary")
    logger.info("="*60)
    logger.info(f"Total samples: {len(features_df)}")
    logger.info(f"Features: {features_df.shape[1]}")
    logger.info(f"Date range: {features_df['match_date'].min()} to {features_df['match_date'].max()}")
    logger.info(f"\nOutcome distribution:")
    logger.info(features_df['outcome'].value_counts())
    logger.info(f"\nGoal distribution:")
    logger.info(f"   Home goals: {features_df['home_goals'].mean():.2f} ± {features_df['home_goals'].std():.2f}")
    logger.info(f"   Away goals: {features_df['away_goals'].mean():.2f} ± {features_df['away_goals'].std():.2f}")
    
    # 8. Show sample features
    logger.info(f"\nSample features (first 5 rows):")
    feature_cols = [c for c in features_df.columns if c not in ['match_date', 'season', 'outcome', 'home_goals', 'away_goals', 'goal_diff', 'gameweek']]
    logger.info(f"Feature columns: {feature_cols[:10]}...")
    
    return features_df


if __name__ == "__main__":
    df = build_training_dataset()
    print(f"\n✓ Training dataset ready with {len(df)} samples")
