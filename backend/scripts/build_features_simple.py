"""Simple feature engineering directly from match dataframe"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def calculate_form_features(team, matches_df, as_of_date, last_n=5, is_home=True):
    """Calculate form features for a team"""
    
    # Get team's matches before the current date
    if is_home:
        team_matches = matches_df[
            (matches_df['home_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        team_matches['goals_for'] = team_matches['home_goals']
        team_matches['goals_against'] = team_matches['away_goals']
    else:
        team_matches = matches_df[
            (matches_df['away_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        team_matches['goals_for'] = team_matches['away_goals']
        team_matches['goals_against'] = team_matches['home_goals']
    
    # Sort by date and take last N
    team_matches = team_matches.sort_values('match_date').tail(last_n)
    
    if len(team_matches) == 0:
        return {
            f'form_last{last_n}': 0,
            f'ppg_last{last_n}': 0,
            f'goals_for_last{last_n}': 0,
            f'goals_against_last{last_n}': 0,
            f'wins_last{last_n}': 0,
            f'draws_last{last_n}': 0,
            f'losses_last{last_n}': 0
        }
    
    # Calculate results
    team_matches['result'] = team_matches.apply(
        lambda x: 'W' if x['goals_for'] > x['goals_against'] else ('D' if x['goals_for'] == x['goals_against'] else 'L'),
        axis=1
    )
    team_matches['points'] = team_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
    
    return {
        f'form_last{last_n}': team_matches['points'].sum(),
        f'ppg_last{last_n}': team_matches['points'].mean(),
        f'goals_for_last{last_n}': team_matches['goals_for'].sum(),
        f'goals_against_last{last_n}': team_matches['goals_against'].sum(),
        f'wins_last{last_n}': (team_matches['result'] == 'W').sum(),
        f'draws_last{last_n}': (team_matches['result'] == 'D').sum(),
        f'losses_last{last_n}': (team_matches['result'] == 'L').sum()
    }


def engineer_match_features(match, all_matches_df, idx):
    """Engineer features for a single match"""
    
    # Only use matches before this one
    historical_matches = all_matches_df[:idx].copy() if idx > 0 else pd.DataFrame()
    
    if historical_matches.empty or len(historical_matches) < 10:
        return None  # Skip early matches
    
    features = {}
    
    # Home team features
    home_form_5 = calculate_form_features(match['home_team'], historical_matches, match['match_date'], last_n=5, is_home=True)
    home_form_10 = calculate_form_features(match['home_team'], historical_matches, match['match_date'], last_n=10, is_home=True)
    
    for key, val in home_form_5.items():
        features[f'home_{key}'] = val
    for key, val in home_form_10.items():
        features[f'home_{key}'] = val
    
    # Away team features
    away_form_5 = calculate_form_features(match['away_team'], historical_matches, match['match_date'], last_n=5, is_home=False)
    away_form_10 = calculate_form_features(match['away_team'], historical_matches, match['match_date'], last_n=10, is_home=False)
    
    for key, val in away_form_5.items():
        features[f'away_{key}'] = val
    for key, val in away_form_10.items():
        features[f'away_{key}'] = val
    
    # Head-to-head features
    h2h = historical_matches[
        ((historical_matches['home_team'] == match['home_team']) & (historical_matches['away_team'] == match['away_team'])) |
        ((historical_matches['home_team'] == match['away_team']) & (historical_matches['away_team'] == match['home_team']))
    ].tail(5)
    
    features['h2h_matches'] = len(h2h)
    if len(h2h) > 0:
        home_h2h_wins = ((h2h['home_team'] == match['home_team']) & (h2h['home_goals'] > h2h['away_goals'])).sum()
        away_h2h_wins = ((h2h['away_team'] == match['home_team']) & (h2h['away_goals'] > h2h['home_goals'])).sum()
        features['h2h_home_wins'] = home_h2h_wins + away_h2h_wins
        features['h2h_draws'] = (h2h['home_goals'] == h2h['away_goals']).sum()
    else:
        features['h2h_home_wins'] = 0
        features['h2h_draws'] = 0
    
    # Form differentials
    features['form_diff_last5'] = features['home_form_last5'] - features['away_form_last5']
    features['goals_for_diff_last5'] = features['home_goals_for_last5'] - features['away_goals_for_last5']
    features['goals_against_diff_last5'] = features['away_goals_against_last5'] - features['home_goals_against_last5']
    
    return features


# Main script
logger.info("="*60)
logger.info("Building Training Dataset with Feature Engineering")
logger.info("="*60)

# Load match data
logger.info("\n1. Loading match data...")
matches_df = pd.read_csv('data/raw/matches_combined.csv')
matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
matches_df = matches_df.sort_values('match_date').reset_index(drop=True)
logger.info(f"   Loaded {len(matches_df)} matches")

# Engineer features
logger.info("\n2. Engineering features...")
features_list = []

for idx, match in matches_df.iterrows():
    if idx % 50 == 0:
        logger.info(f"   Processing match {idx+1}/{len(matches_df)}")
    
    try:
        features = engineer_match_features(match, matches_df, idx)
        
        if features is not None:
            # Add target variables
            features['home_goals'] = match['home_goals']
            features['away_goals'] = match['away_goals']
            features['goal_diff'] = match['home_goals'] - match['away_goals']
            
            # Create outcome target
            if match['home_goals'] > match['away_goals']:
                features['outcome'] = 0  # Home win
            elif match['home_goals'] < match['away_goals']:
                features['outcome'] = 2  # Away win
            else:
                features['outcome'] = 1  # Draw
            
            features['match_date'] = match['match_date']
            features['gameweek'] = match.get('gameweek', 0)
            features['season'] = match.get('season', '')
            features['home_team'] = match['home_team']
            features['away_team'] = match['away_team']
            
            features_list.append(features)
    except Exception as e:
        logger.warning(f"   Error processing match {idx}: {e}")
        continue

# Create DataFrame
logger.info(f"\n3. Creating feature dataset...")
features_df = pd.DataFrame(features_list)
logger.info(f"   Final dataset: {len(features_df)} matches with {features_df.shape[1]} features")

# Save dataset
from pathlib import Path
Path('data/processed').mkdir(parents=True, exist_ok=True)
output_path = 'data/processed/training_dataset.csv'
features_df.to_csv(output_path, index=False)
logger.info(f"\n✓ Saved training dataset to {output_path}")

# Summary
logger.info("\n" + "="*60)
logger.info("Dataset Summary")
logger.info("="*60)
logger.info(f"Total samples: {len(features_df)}")
logger.info(f"Features: {features_df.shape[1]}")
logger.info(f"Date range: {features_df['match_date'].min()} to {features_df['match_date'].max()}")
logger.info(f"\nOutcome distribution:")
logger.info(f"  Home wins (0): {(features_df['outcome'] == 0).sum()}")
logger.info(f"  Draws (1): {(features_df['outcome'] == 1).sum()}")
logger.info(f"  Away wins (2): {(features_df['outcome'] == 2).sum()}")
logger.info(f"\nGoal statistics:")
logger.info(f"  Home goals: {features_df['home_goals'].mean():.2f} ± {features_df['home_goals'].std():.2f}")
logger.info(f"  Away goals: {features_df['away_goals'].mean():.2f} ± {features_df['away_goals'].std():.2f}")

logger.info(f"\n✓ Training dataset ready with {len(features_df)} samples")
