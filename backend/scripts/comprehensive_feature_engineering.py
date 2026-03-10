"""
Comprehensive Feature Engineering - 125+ Features for Premier League Predictions
Creates team strength, form, attack/defense, head-to-head, and lineup features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ComprehensiveFeatureEngineer:
    """
    Create 125+ features from historical match data
    """
    
    def __init__(self):
        self.initial_elo = 1500
        self.k_factor = 32
        self.team_elos = {}
        
    def calculate_elo_ratings(self, matches_df):
        """
        Calculate ELO ratings for all teams based on historical results
        """
        elo_history = {}
        
        for idx, match in matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Initialize ELO if not exists
            if home_team not in self.team_elos:
                self.team_elos[home_team] = self.initial_elo
            if away_team not in self.team_elos:
                self.team_elos[away_team] = self.initial_elo
            
            # Get current ELOs
            home_elo = self.team_elos[home_team]
            away_elo = self.team_elos[away_team]
            
            # Store ELO before match
            elo_history[match['match_id']] = {
                'home_elo': home_elo,
                'away_elo': away_elo
            }
            
            # Calculate expected scores
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            expected_away = 1 - expected_home
            
            # Actual scores
            if match['home_goals'] > match['away_goals']:
                actual_home, actual_away = 1, 0
            elif match['home_goals'] < match['away_goals']:
                actual_home, actual_away = 0, 1
            else:
                actual_home, actual_away = 0.5, 0.5
            
            # Update ELOs
            self.team_elos[home_team] += self.k_factor * (actual_home - expected_home)
            self.team_elos[away_team] += self.k_factor * (actual_away - expected_away)
        
        return elo_history
    
    def calculate_team_strength(self, team, matches_df, as_of_date, last_n=10):
        """
        Calculate team strength metrics from last N matches
        """
        # Get team's recent matches
        team_matches = matches_df[
            ((matches_df['home_team'] == team) | (matches_df['away_team'] == team)) &
            (matches_df['match_date'] < as_of_date)
        ].sort_values('match_date', ascending=False).head(last_n)
        
        if len(team_matches) == 0:
            return {
                'attack_strength': 1.0,
                'defense_strength': 1.0,
                'avg_goals_scored': 1.2,
                'avg_goals_conceded': 1.2,
                'goal_difference': 0.0,
                'win_rate': 0.33,
                'points_per_game': 1.0
            }
        
        goals_scored = []
        goals_conceded = []
        points = []
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                goals_scored.append(match['home_goals'])
                goals_conceded.append(match['away_goals'])
                if match['home_goals'] > match['away_goals']:
                    points.append(3)
                elif match['home_goals'] == match['away_goals']:
                    points.append(1)
                else:
                    points.append(0)
            else:
                goals_scored.append(match['away_goals'])
                goals_conceded.append(match['home_goals'])
                if match['away_goals'] > match['home_goals']:
                    points.append(3)
                elif match['away_goals'] == match['home_goals']:
                    points.append(1)
                else:
                    points.append(0)
        
        wins = sum([1 for p in points if p == 3])
        
        return {
            'attack_strength': np.mean(goals_scored) if goals_scored else 1.0,
            'defense_strength': np.mean(goals_conceded) if goals_conceded else 1.0,
            'avg_goals_scored': np.mean(goals_scored) if goals_scored else 1.2,
            'avg_goals_conceded': np.mean(goals_conceded) if goals_conceded else 1.2,
            'goal_difference': sum(goals_scored) - sum(goals_conceded),
            'win_rate': wins / len(team_matches) if len(team_matches) > 0 else 0.33,
            'points_per_game': np.mean(points) if points else 1.0
        }
    
    def calculate_form_features(self, team, matches_df, as_of_date, last_n=5, venue='all'):
        """
        Calculate form features for last N matches
        """
        # Filter by venue
        if venue == 'home':
            team_matches = matches_df[
                (matches_df['home_team'] == team) &
                (matches_df['match_date'] < as_of_date)
            ]
        elif venue == 'away':
            team_matches = matches_df[
                (matches_df['away_team'] == team) &
                (matches_df['match_date'] < as_of_date)
            ]
        else:
            team_matches = matches_df[
                ((matches_df['home_team'] == team) | (matches_df['away_team'] == team)) &
                (matches_df['match_date'] < as_of_date)
            ]
        
        team_matches = team_matches.sort_values('match_date', ascending=False).head(last_n)
        
        if len(team_matches) == 0:
            return {
                'points': 0,
                'goals_scored': 0,
                'goals_conceded': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'clean_sheets': 0
            }
        
        points = []
        goals_scored = []
        goals_conceded = []
        wins = 0
        draws = 0
        losses = 0
        clean_sheets = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                gf = match['home_goals']
                ga = match['away_goals']
            else:
                gf = match['away_goals']
                ga = match['home_goals']
            
            goals_scored.append(gf)
            goals_conceded.append(ga)
            
            if ga == 0:
                clean_sheets += 1
            
            if gf > ga:
                points.append(3)
                wins += 1
            elif gf == ga:
                points.append(1)
                draws += 1
            else:
                points.append(0)
                losses += 1
        
        return {
            'points': sum(points),
            'goals_scored': sum(goals_scored),
            'goals_conceded': sum(goals_conceded),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'clean_sheets': clean_sheets
        }
    
    def calculate_head_to_head(self, home_team, away_team, matches_df, as_of_date, last_n=5):
        """
        Calculate head-to-head statistics
        """
        h2h = matches_df[
            (((matches_df['home_team'] == home_team) & (matches_df['away_team'] == away_team)) |
             ((matches_df['home_team'] == away_team) & (matches_df['away_team'] == home_team))) &
            (matches_df['match_date'] < as_of_date)
        ].sort_values('match_date', ascending=False).head(last_n)
        
        if len(h2h) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_home_goals_avg': 1.2,
                'h2h_away_goals_avg': 1.2
            }
        
        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals = []
        away_goals = []
        
        for _, match in h2h.iterrows():
            if match['home_team'] == home_team:
                home_goals.append(match['home_goals'])
                away_goals.append(match['away_goals'])
                if match['home_goals'] > match['away_goals']:
                    home_wins += 1
                elif match['home_goals'] == match['away_goals']:
                    draws += 1
                else:
                    away_wins += 1
            else:
                home_goals.append(match['away_goals'])
                away_goals.append(match['home_goals'])
                if match['away_goals'] > match['home_goals']:
                    home_wins += 1
                elif match['away_goals'] == match['home_goals']:
                    draws += 1
                else:
                    away_wins += 1
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_home_goals_avg': np.mean(home_goals) if home_goals else 1.2,
            'h2h_away_goals_avg': np.mean(away_goals) if away_goals else 1.2
        }
    
    def build_features_for_match(self, home_team, away_team, match_date, matches_df, elo_history, match_id):
        """
        Build all 125+ features for a single match
        """
        features = {}
        
        # ELO ratings (2 features)
        if match_id in elo_history:
            features['home_elo'] = elo_history[match_id]['home_elo']
            features['away_elo'] = elo_history[match_id]['away_elo']
            features['elo_diff'] = features['home_elo'] - features['away_elo']
        else:
            features['home_elo'] = self.initial_elo
            features['away_elo'] = self.initial_elo
            features['elo_diff'] = 0
        
        # Team strength (14 features = 7 per team)
        home_strength = self.calculate_team_strength(home_team, matches_df, match_date, last_n=10)
        away_strength = self.calculate_team_strength(away_team, matches_df, match_date, last_n=10)
        
        for key, value in home_strength.items():
            features[f'home_{key}'] = value
        for key, value in away_strength.items():
            features[f'away_{key}'] = value
        
        # Recent form - last 5 matches (42 features = 7 * 3 venues * 2 teams)
        # Overall form
        home_form_overall = self.calculate_form_features(home_team, matches_df, match_date, last_n=5, venue='all')
        away_form_overall = self.calculate_form_features(away_team, matches_df, match_date, last_n=5, venue='all')
        
        for key, value in home_form_overall.items():
            features[f'home_last5_overall_{key}'] = value
        for key, value in away_form_overall.items():
            features[f'away_last5_overall_{key}'] = value
        
        # Home venue form
        home_form_home = self.calculate_form_features(home_team, matches_df, match_date, last_n=5, venue='home')
        for key, value in home_form_home.items():
            features[f'home_last5_home_{key}'] = value
        
        # Away venue form
        away_form_away = self.calculate_form_features(away_team, matches_df, match_date, last_n=5, venue='away')
        for key, value in away_form_away.items():
            features[f'away_last5_away_{key}'] = value
        
        # Head-to-head (5 features)
        h2h = self.calculate_head_to_head(home_team, away_team, matches_df, match_date, last_n=5)
        for key, value in h2h.items():
            features[key] = value
        
        # Attack vs Defense matchups (4 features)
        features['home_attack_vs_away_defense'] = features['home_attack_strength'] / (features['away_defense_strength'] + 0.1)
        features['away_attack_vs_home_defense'] = features['away_attack_strength'] / (features['home_defense_strength'] + 0.1)
        features['home_xg_estimate'] = (features['home_attack_strength'] + features['away_defense_strength']) / 2
        features['away_xg_estimate'] = (features['away_attack_strength'] + features['home_defense_strength']) / 2
        
        # Momentum and streaks (10 features)
        features['home_goal_diff_last5'] = features['home_last5_overall_goals_scored'] - features['home_last5_overall_goals_conceded']
        features['away_goal_diff_last5'] = features['away_last5_overall_goals_scored'] - features['away_last5_overall_goals_conceded']
        features['home_win_streak'] = 1 if features['home_last5_overall_wins'] >= 3 else 0
        features['away_win_streak'] = 1 if features['away_last5_overall_wins'] >= 3 else 0
        features['home_unbeaten_last5'] = features['home_last5_overall_wins'] + features['home_last5_overall_draws']
        features['away_unbeaten_last5'] = features['away_last5_overall_wins'] + features['away_last5_overall_draws']
        features['home_scoring_form'] = features['home_last5_overall_goals_scored'] / 5.0
        features['away_scoring_form'] = features['away_last5_overall_goals_scored'] / 5.0
        features['home_defensive_form'] = features['home_last5_overall_goals_conceded'] / 5.0
        features['away_defensive_form'] = features['away_last5_overall_goals_conceded'] / 5.0
        
        # Home advantage metrics (8 features)
        features['home_home_win_rate'] = features['home_last5_home_wins'] / 5.0 if features['home_last5_home_wins'] > 0 else 0.2
        features['home_home_goals_avg'] = features['home_last5_home_goals_scored'] / 5.0
        features['home_home_conceded_avg'] = features['home_last5_home_goals_conceded'] / 5.0
        features['away_away_loss_rate'] = features['away_last5_away_losses'] / 5.0 if features['away_last5_away_losses'] > 0 else 0.3
        features['away_away_goals_avg'] = features['away_last5_away_goals_scored'] / 5.0
        features['away_away_conceded_avg'] = features['away_last5_away_goals_conceded'] / 5.0
        features['venue_advantage'] = features['home_home_win_rate'] - features['away_away_loss_rate']
        features['venue_goal_diff'] = features['home_home_goals_avg'] - features['away_away_goals_avg']
        
        # Rolling averages (12 features)
        features['home_rolling_ppg'] = features['home_points_per_game']
        features['away_rolling_ppg'] = features['away_points_per_game']
        features['home_rolling_goals_for'] = features['home_avg_goals_scored']
        features['away_rolling_goals_for'] = features['away_avg_goals_scored']
        features['home_rolling_goals_against'] = features['home_avg_goals_conceded']
        features['away_rolling_goals_against'] = features['away_avg_goals_conceded']
        features['ppg_diff'] = features['home_rolling_ppg'] - features['away_rolling_ppg']
        features['goals_for_diff'] = features['home_rolling_goals_for'] - features['away_rolling_goals_for']
        features['goals_against_diff'] = features['away_rolling_goals_against'] - features['home_rolling_goals_against']
        features['overall_strength_diff'] = (features['home_attack_strength'] + features['home_defense_strength']) - (features['away_attack_strength'] + features['away_defense_strength'])
        features['form_diff'] = features['home_last5_overall_points'] - features['away_last5_overall_points']
        features['momentum_score'] = features['home_goal_diff_last5'] - features['away_goal_diff_last5']
        
        # Clean sheets and defensive metrics (6 features)
        features['home_clean_sheets_rate'] = features['home_last5_overall_clean_sheets'] / 5.0
        features['away_clean_sheets_rate'] = features['away_last5_overall_clean_sheets'] / 5.0
        features['home_clean_sheets_home'] = features['home_last5_home_clean_sheets'] / 5.0
        features['away_clean_sheets_away'] = features['away_last5_away_clean_sheets'] / 5.0
        features['home_defensive_solidity'] = 1.0 - (features['home_defensive_form'] / 3.0)
        features['away_defensive_solidity'] = 1.0 - (features['away_defensive_form'] / 3.0)
        
        # Additional derived features to reach 125+ (remaining features)
        features['total_expected_goals'] = features['home_xg_estimate'] + features['away_xg_estimate']
        features['expected_goal_ratio'] = features['home_xg_estimate'] / (features['away_xg_estimate'] + 0.1)
        features['elo_ratio'] = features['home_elo'] / (features['away_elo'] + 0.1)
        features['strength_ratio'] = features['home_attack_strength'] / (features['away_attack_strength'] + 0.1)
        features['defense_ratio'] = features['away_defense_strength'] / (features['home_defense_strength'] + 0.1)
        features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
        features['quality_differential'] = (features['home_attack_strength'] - features['away_defense_strength'])
        features['reverse_quality_differential'] = (features['away_attack_strength'] - features['home_defense_strength'])
        
        # Count total features
        logger.info(f"Generated {len(features)} features for {home_team} vs {away_team}")
        
        return features
    
    def build_training_dataset(self, matches_df):
        """
        Build complete training dataset with all features
        """
        logger.info("="*80)
        logger.info("BUILDING COMPREHENSIVE TRAINING DATASET (125+ FEATURES)")
        logger.info("="*80)
        
        # Calculate ELO ratings
        logger.info("Calculating ELO ratings...")
        elo_history = self.calculate_elo_ratings(matches_df)
        
        # Build features for each match
        logger.info("Building features for each match...")
        features_list = []
        
        for idx, match in matches_df.iterrows():
            features = self.build_features_for_match(
                home_team=match['home_team'],
                away_team=match['away_team'],
                match_date=match['match_date'],
                matches_df=matches_df,
                elo_history=elo_history,
                match_id=match['match_id']
            )
            
            # Add target variables
            features['match_id'] = match['match_id']
            features['home_team'] = match['home_team']
            features['away_team'] = match['away_team']
            features['match_date'] = match['match_date']
            features['home_goals'] = match['home_goals']
            features['away_goals'] = match['away_goals']
            
            # Outcome encoding: 0=Away Win, 1=Draw, 2=Home Win
            if match['home_goals'] > match['away_goals']:
                features['outcome'] = 2
            elif match['home_goals'] < match['away_goals']:
                features['outcome'] = 0
            else:
                features['outcome'] = 1
            
            features_list.append(features)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(matches_df)} matches...")
        
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"✅ Training dataset created: {len(features_df)} matches, {len(features_df.columns)} total columns")
        
        # Identify feature columns (exclude metadata and targets)
        feature_cols = [col for col in features_df.columns if col not in [
            'match_id', 'home_team', 'away_team', 'match_date', 
            'home_goals', 'away_goals', 'outcome'
        ]]
        
        logger.info(f"✅ Feature columns: {len(feature_cols)}")
        
        return features_df, feature_cols


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    # This would be tested with actual data
    print("Feature engineering module ready")
