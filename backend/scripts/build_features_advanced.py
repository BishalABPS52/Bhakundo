"""Advanced feature engineering for improved draw and away win prediction"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def calculate_form_features(team, matches_df, as_of_date, last_n=5, venue='all'):
    """Calculate form features for a team
    
    Args:
        venue: 'all', 'home', or 'away'
    """
    
    # Get team's matches before the current date
    if venue == 'home':
        team_matches = matches_df[
            (matches_df['home_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        team_matches['goals_for'] = team_matches['home_goals']
        team_matches['goals_against'] = team_matches['away_goals']
    elif venue == 'away':
        team_matches = matches_df[
            (matches_df['away_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        team_matches['goals_for'] = team_matches['away_goals']
        team_matches['goals_against'] = team_matches['home_goals']
    else:  # all
        home_matches = matches_df[
            (matches_df['home_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        home_matches['goals_for'] = home_matches['home_goals']
        home_matches['goals_against'] = home_matches['away_goals']
        
        away_matches = matches_df[
            (matches_df['away_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        away_matches['goals_for'] = away_matches['away_goals']
        away_matches['goals_against'] = away_matches['home_goals']
        
        team_matches = pd.concat([home_matches, away_matches])
    
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
            f'losses_last{last_n}': 0,
            f'clean_sheets_last{last_n}': 0,
            f'btts_last{last_n}': 0,  # Both teams to score
            f'over25_last{last_n}': 0  # Over 2.5 goals
        }
    
    # Calculate results
    team_matches['result'] = team_matches.apply(
        lambda x: 'W' if x['goals_for'] > x['goals_against'] else ('D' if x['goals_for'] == x['goals_against'] else 'L'),
        axis=1
    )
    team_matches['points'] = team_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
    team_matches['clean_sheet'] = (team_matches['goals_against'] == 0).astype(int)
    team_matches['btts'] = ((team_matches['goals_for'] > 0) & (team_matches['goals_against'] > 0)).astype(int)
    team_matches['over25'] = ((team_matches['goals_for'] + team_matches['goals_against']) > 2.5).astype(int)
    
    return {
        f'form_last{last_n}': team_matches['points'].sum(),
        f'ppg_last{last_n}': team_matches['points'].mean(),
        f'goals_for_last{last_n}': team_matches['goals_for'].sum(),
        f'goals_against_last{last_n}': team_matches['goals_against'].sum(),
        f'wins_last{last_n}': (team_matches['result'] == 'W').sum(),
        f'draws_last{last_n}': (team_matches['result'] == 'D').sum(),
        f'losses_last{last_n}': (team_matches['result'] == 'L').sum(),
        f'clean_sheets_last{last_n}': team_matches['clean_sheet'].sum(),
        f'btts_last{last_n}': team_matches['btts'].sum(),
        f'over25_last{last_n}': team_matches['over25'].sum()
    }


def calculate_momentum_features(team, matches_df, as_of_date, venue='all'):
    """Calculate momentum and trend features"""
    
    if venue == 'home':
        team_matches = matches_df[
            (matches_df['home_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        team_matches['goals_for'] = team_matches['home_goals']
        team_matches['goals_against'] = team_matches['away_goals']
    elif venue == 'away':
        team_matches = matches_df[
            (matches_df['away_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        team_matches['goals_for'] = team_matches['away_goals']
        team_matches['goals_against'] = team_matches['home_goals']
    else:
        home_matches = matches_df[
            (matches_df['home_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        home_matches['goals_for'] = home_matches['home_goals']
        home_matches['goals_against'] = home_matches['away_goals']
        
        away_matches = matches_df[
            (matches_df['away_team'] == team) & 
            (matches_df['match_date'] < as_of_date)
        ].copy()
        away_matches['goals_for'] = away_matches['away_goals']
        away_matches['goals_against'] = away_matches['home_goals']
        
        team_matches = pd.concat([home_matches, away_matches])
    
    team_matches = team_matches.sort_values('match_date').tail(10)
    
    if len(team_matches) < 5:
        return {
            'momentum_recent': 0,
            'trend_points': 0,
            'consistency': 0,
            'avg_goal_diff': 0
        }
    
    team_matches['result'] = team_matches.apply(
        lambda x: 'W' if x['goals_for'] > x['goals_against'] else ('D' if x['goals_for'] == x['goals_against'] else 'L'),
        axis=1
    )
    team_matches['points'] = team_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
    team_matches['goal_diff'] = team_matches['goals_for'] - team_matches['goals_against']
    
    # Weighted momentum (recent matches matter more)
    weights = np.linspace(0.5, 1.5, len(team_matches))
    momentum = np.sum(team_matches['points'].values * weights) / np.sum(weights)
    
    # Trend: compare first half vs second half
    mid = len(team_matches) // 2
    first_half_ppg = team_matches.iloc[:mid]['points'].mean()
    second_half_ppg = team_matches.iloc[mid:]['points'].mean()
    trend = second_half_ppg - first_half_ppg
    
    # Consistency: std of points
    consistency = team_matches['points'].std()
    
    return {
        'momentum_recent': momentum,
        'trend_points': trend,
        'consistency': consistency,
        'avg_goal_diff': team_matches['goal_diff'].mean()
    }


def calculate_rest_days(team, matches_df, match_date, is_home=True):
    """Calculate days since last match"""
    
    if is_home:
        prev_matches = matches_df[
            ((matches_df['home_team'] == team) | (matches_df['away_team'] == team)) &
            (matches_df['match_date'] < match_date)
        ]
    else:
        prev_matches = matches_df[
            ((matches_df['home_team'] == team) | (matches_df['away_team'] == team)) &
            (matches_df['match_date'] < match_date)
        ]
    
    if len(prev_matches) == 0:
        return 7  # Default
    
    last_match = prev_matches.sort_values('match_date').iloc[-1]
    rest_days = (match_date - last_match['match_date']).days
    
    return rest_days


def engineer_match_features(match, all_matches_df, idx):
    """Engineer comprehensive features for a single match"""
    
    # Only use matches before this one
    historical_matches = all_matches_df[:idx].copy() if idx > 0 else pd.DataFrame()
    
    if historical_matches.empty or len(historical_matches) < 10:
        return None  # Skip early matches
    
    features = {}
    
    # ========== HOME TEAM FEATURES ==========
    
    # Overall form (last 5 and 10 matches)
    home_form_5 = calculate_form_features(match['home_team'], historical_matches, match['match_date'], last_n=5, venue='all')
    home_form_10 = calculate_form_features(match['home_team'], historical_matches, match['match_date'], last_n=10, venue='all')
    
    for key, val in home_form_5.items():
        features[f'home_{key}'] = val
    for key, val in home_form_10.items():
        features[f'home_{key}'] = val
    
    # Venue-specific form (home matches only)
    home_venue_5 = calculate_form_features(match['home_team'], historical_matches, match['match_date'], last_n=5, venue='home')
    for key, val in home_venue_5.items():
        features[f'home_venue_{key}'] = val
    
    # Momentum features
    home_momentum = calculate_momentum_features(match['home_team'], historical_matches, match['match_date'], venue='all')
    for key, val in home_momentum.items():
        features[f'home_{key}'] = val
    
    # Rest days
    features['home_rest_days'] = calculate_rest_days(match['home_team'], historical_matches, match['match_date'], is_home=True)
    
    
    # ========== AWAY TEAM FEATURES ==========
    
    # Overall form (last 5 and 10 matches)
    away_form_5 = calculate_form_features(match['away_team'], historical_matches, match['match_date'], last_n=5, venue='all')
    away_form_10 = calculate_form_features(match['away_team'], historical_matches, match['match_date'], last_n=10, venue='all')
    
    for key, val in away_form_5.items():
        features[f'away_{key}'] = val
    for key, val in away_form_10.items():
        features[f'away_{key}'] = val
    
    # Venue-specific form (away matches only)
    away_venue_5 = calculate_form_features(match['away_team'], historical_matches, match['match_date'], last_n=5, venue='away')
    for key, val in away_venue_5.items():
        features[f'away_venue_{key}'] = val
    
    # Momentum features
    away_momentum = calculate_momentum_features(match['away_team'], historical_matches, match['match_date'], venue='all')
    for key, val in away_momentum.items():
        features[f'away_{key}'] = val
    
    # Rest days
    features['away_rest_days'] = calculate_rest_days(match['away_team'], historical_matches, match['match_date'], is_home=False)
    
    
    # ========== HEAD-TO-HEAD FEATURES ==========
    
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
        features['h2h_avg_goals'] = (h2h['home_goals'] + h2h['away_goals']).mean()
    else:
        features['h2h_home_wins'] = 0
        features['h2h_draws'] = 0
        features['h2h_avg_goals'] = 2.5  # League average
    
    
    # ========== COMPARATIVE FEATURES ==========
    
    # Form differentials
    features['form_diff_last5'] = features['home_form_last5'] - features['away_form_last5']
    features['ppg_diff_last5'] = features['home_ppg_last5'] - features['away_ppg_last5']
    features['goals_for_diff_last5'] = features['home_goals_for_last5'] - features['away_goals_for_last5']
    features['goals_against_diff_last5'] = features['away_goals_against_last5'] - features['home_goals_against_last5']
    
    # Venue-specific differentials
    features['venue_form_diff'] = features['home_venue_form_last5'] - features['away_venue_form_last5']
    features['venue_goals_diff'] = features['home_venue_goals_for_last5'] - features['away_venue_goals_for_last5']
    
    # Momentum differentials
    features['momentum_diff'] = features['home_momentum_recent'] - features['away_momentum_recent']
    features['trend_diff'] = features['home_trend_points'] - features['away_trend_points']
    
    # Rest differential
    features['rest_days_diff'] = features['home_rest_days'] - features['away_rest_days']
    
    # Draw indicators
    features['both_defensive'] = (features['home_goals_against_last5'] < 5) & (features['away_goals_against_last5'] < 5)
    features['evenly_matched'] = abs(features['ppg_diff_last5']) < 0.5
    features['both_drawing_freq'] = (features['home_draws_last5'] + features['away_draws_last5']) / 10.0
    
    return features


# Main script
logger.info("="*70)
logger.info("Building Advanced Training Dataset with Enhanced Features")
logger.info("="*70)

# Load combined matches
logger.info("\n1. Loading match data...")
df = pd.read_csv('data/raw/matches_combined.csv')
df['match_date'] = pd.to_datetime(df['match_date'])
logger.info(f"   Loaded {len(df)} matches")

# Sort by date
df = df.sort_values('match_date').reset_index(drop=True)

# Engineer features
logger.info("\n2. Engineering advanced features...")
features_list = []
targets = []

for idx, row in df.iterrows():
    if idx % 100 == 0:
        logger.info(f"   Processing match {idx+1}/{len(df)}")
    
    features = engineer_match_features(row, df, idx)
    
    if features is not None:
        # Add target variables
        features['home_goals'] = row['home_goals']
        features['away_goals'] = row['away_goals']
        features['goal_diff'] = row['home_goals'] - row['away_goals']
        
        # Outcome: 0=Home win, 1=Draw, 2=Away win
        if row['home_goals'] > row['away_goals']:
            features['outcome'] = 0
        elif row['home_goals'] == row['away_goals']:
            features['outcome'] = 1
        else:
            features['outcome'] = 2
        
        # Metadata
        features['match_date'] = row['match_date']
        features['gameweek'] = row['gameweek']
        features['season'] = row['season']
        features['home_team'] = row['home_team']
        features['away_team'] = row['away_team']
        
        features_list.append(features)

# Create dataframe
logger.info("\n3. Creating dataset...")
training_df = pd.DataFrame(features_list)

# Summary statistics
logger.info("\n4. Dataset Summary:")
logger.info(f"   Total matches: {len(training_df)}")
logger.info(f"   Features: {len([c for c in training_df.columns if c not in ['match_date', 'season', 'home_team', 'away_team', 'outcome', 'home_goals', 'away_goals', 'goal_diff', 'gameweek']])}")
logger.info(f"   Date range: {training_df['match_date'].min()} to {training_df['match_date'].max()}")
logger.info(f"\n   Target distribution:")
logger.info(f"   Home wins (0): {(training_df['outcome']==0).sum()}")
logger.info(f"   Draws (1): {(training_df['outcome']==1).sum()}")
logger.info(f"   Away wins (2): {(training_df['outcome']==2).sum()}")
logger.info(f"\n   Goals statistics:")
logger.info(f"   Home goals: {training_df['home_goals'].mean():.2f} ± {training_df['home_goals'].std():.2f}")
logger.info(f"   Away goals: {training_df['away_goals'].mean():.2f} ± {training_df['away_goals'].std():.2f}")

# Save dataset
output_path = 'data/processed/training_dataset_advanced.csv'
training_df.to_csv(output_path, index=False)
logger.info(f"\n✓ Saved to {output_path}")
logger.info("="*70)
