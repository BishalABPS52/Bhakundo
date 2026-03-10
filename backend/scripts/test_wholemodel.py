"""
Train Score Prediction Model
Uses XGBoost to predict exact match scores based on team statistics and historical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_team_stats(df, team, home=True, is_home_match=True):
    """Calculate comprehensive team statistics with venue-specific metrics"""
    prefix = 'home' if home else 'away'
    team_col = 'home_team' if home else 'away_team'
    goals_for_col = 'home_goals' if home else 'away_goals'
    goals_against_col = 'away_goals' if home else 'home_goals'
    
    # Filter ALL team matches (home and away)
    all_team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    
    # Filter venue-specific matches (home or away)
    if is_home_match:
        venue_matches = df[df['home_team'] == team].copy()
    else:
        venue_matches = df[df['away_team'] == team].copy()
    
    if len(all_team_matches) < 3:
        # Not enough data, return defaults
        return {
            f'{prefix}_avg_goals_scored_l5': 1.5,
            f'{prefix}_avg_goals_conceded_l5': 1.5,
            f'{prefix}_avg_goals_scored_l10': 1.5,
            f'{prefix}_avg_goals_conceded_l10': 1.5,
            f'{prefix}_avg_goals_scored_l20': 1.5,
            f'{prefix}_avg_goals_conceded_l20': 1.5,
            f'{prefix}_ppg_l5': 1.5,
            f'{prefix}_ppg_l10': 1.5,
            f'{prefix}_ppg_l20': 1.5,
            f'{prefix}_win_rate_l5': 0.33,
            f'{prefix}_win_rate_l10': 0.33,
            f'{prefix}_clean_sheets_l5': 0.2,
            f'{prefix}_clean_sheets_l10': 0.2,
            f'{prefix}_btts_l5': 0.5,
            f'{prefix}_btts_l10': 0.5,
            f'{prefix}_scoring_form': 1.5,
            f'{prefix}_defensive_form': 1.5,
            f'{prefix}_xg_estimate': 1.5,
            f'{prefix}_xga_estimate': 1.5,
            f'{prefix}_venue_strength': 0.5,
            f'{prefix}_high_scoring_rate': 0.3,
            f'{prefix}_low_scoring_rate': 0.3,
            f'{prefix}_goals_variance': 1.0,
            f'{prefix}_scoring_consistency': 0.5,
            f'{prefix}_recent_momentum': 0.0,
        }
    
    # Sort by date
    all_team_matches = all_team_matches.sort_values('match_date')
    
    # Extract goals for all matches
    goals_scored = []
    goals_conceded = []
    points = []
    
    for _, match in all_team_matches.iterrows():
        if match['home_team'] == team:
            gf, ga = match['home_goals'], match['away_goals']
        else:
            gf, ga = match['away_goals'], match['home_goals']
        
        goals_scored.append(gf)
        goals_conceded.append(ga)
        
        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)
    
    goals_scored = np.array(goals_scored)
    goals_conceded = np.array(goals_conceded)
    
    # Calculate rolling averages for different windows
    last_5_gf = goals_scored[-5:].mean() if len(goals_scored) >= 5 else goals_scored.mean()
    last_5_ga = goals_conceded[-5:].mean() if len(goals_conceded) >= 5 else goals_conceded.mean()
    last_10_gf = goals_scored[-10:].mean() if len(goals_scored) >= 10 else goals_scored.mean()
    last_10_ga = goals_conceded[-10:].mean() if len(goals_conceded) >= 10 else goals_conceded.mean()
    last_20_gf = goals_scored[-20:].mean() if len(goals_scored) >= 20 else goals_scored.mean()
    last_20_ga = goals_conceded[-20:].mean() if len(goals_conceded) >= 20 else goals_conceded.mean()
    
    # Points per game
    last_5_points = points[-5:] if len(points) >= 5 else points
    last_10_points = points[-10:] if len(points) >= 10 else points
    last_20_points = points[-20:] if len(points) >= 20 else points
    
    ppg_l5 = np.mean(last_5_points) if last_5_points else 1.5
    ppg_l10 = np.mean(last_10_points) if last_10_points else 1.5
    ppg_l20 = np.mean(last_20_points) if last_20_points else 1.5
    
    # Win rates
    win_rate_l5 = sum(1 for p in last_5_points if p == 3) / len(last_5_points) if last_5_points else 0.33
    win_rate_l10 = sum(1 for p in last_10_points if p == 3) / len(last_10_points) if last_10_points else 0.33
    
    # Clean sheets
    clean_sheets_l5 = sum(1 for ga in goals_conceded[-5:] if ga == 0) / min(5, len(goals_conceded)) if len(goals_conceded) > 0 else 0.2
    clean_sheets_l10 = sum(1 for ga in goals_conceded[-10:] if ga == 0) / min(10, len(goals_conceded)) if len(goals_conceded) > 0 else 0.2
    
    # Both teams to score rate
    btts_l5 = sum(1 for i in range(-5, 0) if -i <= len(goals_scored) and goals_scored[i] > 0 and goals_conceded[i] > 0) / min(5, len(goals_scored)) if len(goals_scored) > 0 else 0.5
    btts_l10 = sum(1 for i in range(-10, 0) if -i <= len(goals_scored) and goals_scored[i] > 0 and goals_conceded[i] > 0) / min(10, len(goals_scored)) if len(goals_scored) > 0 else 0.5
    
    # Scoring form (weighted recent average)
    if len(goals_scored) >= 5:
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # More weight to recent matches
        scoring_form = np.average(goals_scored[-5:], weights=weights)
    else:
        scoring_form = goals_scored.mean() if len(goals_scored) > 0 else 1.5
    
    # Defensive form
    if len(goals_conceded) >= 5:
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        defensive_form = np.average(goals_conceded[-5:], weights=weights)
    else:
        defensive_form = goals_conceded.mean() if len(goals_conceded) > 0 else 1.5
    
    # Expected goals estimate (based on attacking strength and recent form)
    xg_estimate = (last_5_gf * 0.5 + last_10_gf * 0.3 + last_20_gf * 0.2) if len(goals_scored) >= 5 else last_5_gf
    xga_estimate = (last_5_ga * 0.5 + last_10_ga * 0.3 + last_20_ga * 0.2) if len(goals_conceded) >= 5 else last_5_ga
    
    # Venue-specific strength
    if len(venue_matches) > 0:
        venue_goals = []
        for _, match in venue_matches.iterrows():
            if is_home_match:
                venue_goals.append(match['home_goals'])
            else:
                venue_goals.append(match['away_goals'])
        venue_strength = np.mean(venue_goals[-10:]) if len(venue_goals) >= 10 else np.mean(venue_goals)
    else:
        venue_strength = xg_estimate * 0.9  # Slight penalty if no venue data
    
    # High/low scoring match rate
    high_scoring_rate = sum(1 for gf in goals_scored[-10:] if gf >= 3) / min(10, len(goals_scored)) if len(goals_scored) > 0 else 0.3
    low_scoring_rate = sum(1 for gf in goals_scored[-10:] if gf == 0) / min(10, len(goals_scored)) if len(goals_scored) > 0 else 0.3
    
    # Goals variance (consistency metric)
    goals_variance = np.var(goals_scored[-10:]) if len(goals_scored) >= 10 else np.var(goals_scored) if len(goals_scored) > 0 else 1.0
    
    # Scoring consistency (inverse of coefficient of variation)
    if len(goals_scored) >= 5 and goals_scored[-5:].mean() > 0:
        cv = goals_scored[-5:].std() / goals_scored[-5:].mean()
        scoring_consistency = 1 / (1 + cv)  # Normalize between 0 and 1
    else:
        scoring_consistency = 0.5
    
    # Recent momentum (trend in last 5 vs previous 5)
    if len(goals_scored) >= 10:
        recent_5 = goals_scored[-5:].mean()
        previous_5 = goals_scored[-10:-5].mean()
        recent_momentum = recent_5 - previous_5
    else:
        recent_momentum = 0.0
    
    return {
        f'{prefix}_avg_goals_scored_l5': last_5_gf,
        f'{prefix}_avg_goals_conceded_l5': last_5_ga,
        f'{prefix}_avg_goals_scored_l10': last_10_gf,
        f'{prefix}_avg_goals_conceded_l10': last_10_ga,
        f'{prefix}_avg_goals_scored_l20': last_20_gf,
        f'{prefix}_avg_goals_conceded_l20': last_20_ga,
        f'{prefix}_ppg_l5': ppg_l5,
        f'{prefix}_ppg_l10': ppg_l10,
        f'{prefix}_ppg_l20': ppg_l20,
        f'{prefix}_win_rate_l5': win_rate_l5,
        f'{prefix}_win_rate_l10': win_rate_l10,
        f'{prefix}_clean_sheets_l5': clean_sheets_l5,
        f'{prefix}_clean_sheets_l10': clean_sheets_l10,
        f'{prefix}_btts_l5': btts_l5,
        f'{prefix}_btts_l10': btts_l10,
        f'{prefix}_scoring_form': scoring_form,
        f'{prefix}_defensive_form': defensive_form,
        f'{prefix}_xg_estimate': xg_estimate,
        f'{prefix}_xga_estimate': xga_estimate,
        f'{prefix}_venue_strength': venue_strength,
        f'{prefix}_high_scoring_rate': high_scoring_rate,
        f'{prefix}_low_scoring_rate': low_scoring_rate,
        f'{prefix}_goals_variance': goals_variance,
        f'{prefix}_scoring_consistency': scoring_consistency,
        f'{prefix}_recent_momentum': recent_momentum,
    }


def create_features(df):
    """Create feature matrix for score prediction"""
    logger.info("Creating features for score prediction...")
    
    features_list = []
    
    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)
    
    # Build team stats over time
    team_stats_cache = {}
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing match {idx}/{len(df)}")
        
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get historical data up to this match
        historical_df = df.iloc[:idx]
        
        # Get team stats
        home_stats = create_team_stats(historical_df, home_team, home=True, is_home_match=True)
        away_stats = create_team_stats(historical_df, away_team, home=False, is_home_match=False)
        
        # Head to head stats
        h2h = historical_df[
            ((historical_df['home_team'] == home_team) & (historical_df['away_team'] == away_team)) |
            ((historical_df['home_team'] == away_team) & (historical_df['away_team'] == home_team))
        ]
        
        h2h_home_goals = 1.5
        h2h_away_goals = 1.5
        h2h_matches = 0
        h2h_home_wins = 0
        h2h_draws = 0
        h2h_btts_rate = 0.5
        h2h_over_25 = 0.5
        
        if len(h2h) > 0:
            h2h_matches = len(h2h)
            # When home_team was at home against away_team
            h2h_home = h2h[h2h['home_team'] == home_team]
            # When home_team was away against away_team
            h2h_away = h2h[h2h['away_team'] == home_team]
            
            if len(h2h_home) > 0:
                h2h_home_goals = h2h_home['home_goals'].mean()
                h2h_home_wins = (h2h_home['home_goals'] > h2h_home['away_goals']).sum() / len(h2h_home)
            if len(h2h_away) > 0:
                h2h_away_goals = h2h_away['away_goals'].mean()
            
            # Both teams to score in H2H
            h2h_btts_rate = ((h2h['home_goals'] > 0) & (h2h['away_goals'] > 0)).mean()
            
            # Over 2.5 goals in H2H
            h2h_over_25 = ((h2h['home_goals'] + h2h['away_goals']) > 2.5).mean()
        
        # Team strength differential
        home_attack_strength = home_stats['home_xg_estimate']
        away_attack_strength = away_stats['away_xg_estimate']
        home_defense_strength = 3.0 - home_stats['home_xga_estimate']  # Inverse (3 - xGA)
        away_defense_strength = 3.0 - away_stats['away_xga_estimate']
        
        attack_differential = home_attack_strength - away_attack_strength
        defense_differential = home_defense_strength - away_defense_strength
        
        # Match context features
        total_xg = home_stats['home_xg_estimate'] + away_stats['away_xg_estimate']
        expected_goal_diff = home_stats['home_xg_estimate'] - away_stats['away_xg_estimate']
        
        # Form momentum differential
        momentum_diff = home_stats['home_recent_momentum'] - away_stats['away_recent_momentum']
        
        # Scoring tendency
        home_attacking_tendency = home_stats['home_high_scoring_rate']
        away_attacking_tendency = away_stats['away_high_scoring_rate']
        defensive_match_likelihood = (home_stats['home_low_scoring_rate'] + away_stats['away_low_scoring_rate']) / 2
        
        # Create feature dict
        features = {
            **home_stats,
            **away_stats,
            'h2h_matches': h2h_matches,
            'h2h_home_goals_avg': h2h_home_goals,
            'h2h_away_goals_avg': h2h_away_goals,
            'h2h_home_win_rate': h2h_home_wins,
            'h2h_draw_rate': h2h_draws,
            'h2h_btts_rate': h2h_btts_rate,
            'h2h_over_25_rate': h2h_over_25,
            'attack_differential': attack_differential,
            'defense_differential': defense_differential,
            'total_xg': total_xg,
            'expected_goal_diff': expected_goal_diff,
            'momentum_diff': momentum_diff,
            'home_attacking_tendency': home_attacking_tendency,
            'away_attacking_tendency': away_attacking_tendency,
            'defensive_match_likelihood': defensive_match_likelihood,
            'home_goals': row['home_goals'],  # Target
            'away_goals': row['away_goals'],  # Target
            'gameweek': row['gameweek'],
            'season': row['season']
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def test_recent_matches():
    """Test model on last 18 matches"""
    logger.info("="*80)
    logger.info("TESTING SCORE PREDICTION MODEL ON RECENT MATCHES")
    logger.info("="*80)
    
    # Load model
    logger.info("Loading trained model...")
    model_data = joblib.load('data/models/score_prediction_model.pkl')
    home_model = model_data['home_model']
    away_model = model_data['away_model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    logger.info(f"Model trained on: {model_data['training_date']}")
    logger.info(f"Training set accuracy: {model_data['exact_score_accuracy']:.1%}")
    
    # Load all matches
    logger.info("\nLoading match data...")
    df = pd.read_csv('data/raw/matches_combined.csv')
    df = df[df['status'] == 'FINISHED'].copy()
    df = df.sort_values('match_date').reset_index(drop=True)
    
    # Get current season matches
    current_season = '2025-26'
    season_matches = df[df['season'] == current_season].copy()
    logger.info(f"Current season ({current_season}): {len(season_matches)} matches")
    
    # Get last 18 matches
    test_matches = season_matches.tail(18).copy()
    logger.info(f"Testing on last 18 matches (Gameweeks: {test_matches['gameweek'].min()}-{test_matches['gameweek'].max()})")
    
    # Prepare predictions
    predictions = []
    
    logger.info("\n" + "="*80)
    logger.info("MATCH-BY-MATCH PREDICTIONS")
    logger.info("="*80)
    
    for idx, row in test_matches.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        actual_home = int(row['home_goals'])
        actual_away = int(row['away_goals'])
        gameweek = row['gameweek']
        
        # Get historical data up to this match (excluding this match)
        historical_df = df[df.index < idx]
        
        # Get team stats
        home_stats = create_team_stats(historical_df, home_team, home=True, is_home_match=True)
        away_stats = create_team_stats(historical_df, away_team, home=False, is_home_match=False)
        
        # Head to head stats
        h2h = historical_df[
            ((historical_df['home_team'] == home_team) & (historical_df['away_team'] == away_team)) |
            ((historical_df['home_team'] == away_team) & (historical_df['away_team'] == home_team))
        ]
        
        h2h_home_goals = 1.5
        h2h_away_goals = 1.5
        h2h_matches = 0
        h2h_home_wins = 0
        h2h_draws = 0
        h2h_btts_rate = 0.5
        h2h_over_25 = 0.5
        
        if len(h2h) > 0:
            h2h_matches = len(h2h)
            h2h_home = h2h[h2h['home_team'] == home_team]
            h2h_away = h2h[h2h['away_team'] == home_team]
            
            if len(h2h_home) > 0:
                h2h_home_goals = h2h_home['home_goals'].mean()
                h2h_home_wins = (h2h_home['home_goals'] > h2h_home['away_goals']).sum() / len(h2h_home)
            if len(h2h_away) > 0:
                h2h_away_goals = h2h_away['away_goals'].mean()
            
            h2h_btts_rate = ((h2h['home_goals'] > 0) & (h2h['away_goals'] > 0)).mean()
            h2h_over_25 = ((h2h['home_goals'] + h2h['away_goals']) > 2.5).mean()
        
        # Team strength differential
        home_attack_strength = home_stats['home_xg_estimate']
        away_attack_strength = away_stats['away_xg_estimate']
        home_defense_strength = 3.0 - home_stats['home_xga_estimate']
        away_defense_strength = 3.0 - away_stats['away_xga_estimate']
        
        attack_differential = home_attack_strength - away_attack_strength
        defense_differential = home_defense_strength - away_defense_strength
        
        # Match context features
        total_xg = home_stats['home_xg_estimate'] + away_stats['away_xg_estimate']
        expected_goal_diff = home_stats['home_xg_estimate'] - away_stats['away_xg_estimate']
        
        # Form momentum differential
        momentum_diff = home_stats['home_recent_momentum'] - away_stats['away_recent_momentum']
        
        # Scoring tendency
        home_attacking_tendency = home_stats['home_high_scoring_rate']
        away_attacking_tendency = away_stats['away_high_scoring_rate']
        defensive_match_likelihood = (home_stats['home_low_scoring_rate'] + away_stats['away_low_scoring_rate']) / 2
        
        # Create feature vector
        features = {
            **home_stats,
            **away_stats,
            'h2h_matches': h2h_matches,
            'h2h_home_goals_avg': h2h_home_goals,
            'h2h_away_goals_avg': h2h_away_goals,
            'h2h_home_win_rate': h2h_home_wins,
            'h2h_draw_rate': h2h_draws,
            'h2h_btts_rate': h2h_btts_rate,
            'h2h_over_25_rate': h2h_over_25,
            'attack_differential': attack_differential,
            'defense_differential': defense_differential,
            'total_xg': total_xg,
            'expected_goal_diff': expected_goal_diff,
            'momentum_diff': momentum_diff,
            'home_attacking_tendency': home_attacking_tendency,
            'away_attacking_tendency': away_attacking_tendency,
            'defensive_match_likelihood': defensive_match_likelihood,
            'gameweek': gameweek,
        }
        
        # Prepare for prediction
        X = pd.DataFrame([features])[feature_columns]
        X_scaled = scaler.transform(X)
        
        # Predict
        pred_home = home_model.predict(X_scaled)[0]
        pred_away = away_model.predict(X_scaled)[0]
        
        pred_home_rounded = int(round(pred_home))
        pred_away_rounded = int(round(pred_away))
        
        # Check accuracy
        exact_match = (pred_home_rounded == actual_home) and (pred_away_rounded == actual_away)
        
        # Check outcome accuracy
        actual_outcome = np.sign(actual_home - actual_away)
        pred_outcome = np.sign(pred_home_rounded - pred_away_rounded)
        correct_outcome = (actual_outcome == pred_outcome)
        
        # Calculate goal difference accuracy
        actual_gd = abs(actual_home - actual_away)
        pred_gd = abs(pred_home_rounded - pred_away_rounded)
        gd_diff = abs(actual_gd - pred_gd)
        
        predictions.append({
            'gameweek': gameweek,
            'home_team': home_team,
            'away_team': away_team,
            'actual_score': f"{actual_home}-{actual_away}",
            'predicted_score': f"{pred_home_rounded}-{pred_away_rounded}",
            'exact_match': exact_match,
            'correct_outcome': correct_outcome,
            'home_goals_diff': abs(pred_home_rounded - actual_home),
            'away_goals_diff': abs(pred_away_rounded - actual_away),
            'gd_diff': gd_diff
        })
        
        # Display result
        status = "✓ EXACT" if exact_match else ("✓ OUTCOME" if correct_outcome else "✗ WRONG")
        logger.info(f"GW{gameweek:2d} | {home_team:25s} {actual_home}-{actual_away} {away_team:25s} | Pred: {pred_home_rounded}-{pred_away_rounded} | {status}")
    
    # Calculate overall accuracy
    results_df = pd.DataFrame(predictions)
    
    exact_score_accuracy = results_df['exact_match'].mean()
    outcome_accuracy = results_df['correct_outcome'].mean()
    avg_home_error = results_df['home_goals_diff'].mean()
    avg_away_error = results_df['away_goals_diff'].mean()
    avg_total_error = (results_df['home_goals_diff'] + results_df['away_goals_diff']).mean()
    
    logger.info("\n" + "="*80)
    logger.info("OVERALL ACCURACY METRICS")
    logger.info("="*80)
    logger.info(f"Exact Score Accuracy:     {exact_score_accuracy:6.1%}  ({int(exact_score_accuracy * 18)}/18 matches)")
    logger.info(f"Correct Outcome:          {outcome_accuracy:6.1%}  ({int(outcome_accuracy * 18)}/18 matches)")
    logger.info(f"Average Home Goals Error: {avg_home_error:6.2f} goals")
    logger.info(f"Average Away Goals Error: {avg_away_error:6.2f} goals")
    logger.info(f"Average Total Error:      {avg_total_error:6.2f} goals per match")
    
    # Show breakdown by outcome
    logger.info("\n" + "="*80)
    logger.info("BREAKDOWN BY ACTUAL RESULT")
    logger.info("="*80)
    
    for match in predictions:
        actual_h, actual_a = map(int, match['actual_score'].split('-'))
        if actual_h > actual_a:
            match['result_type'] = 'Home Win'
        elif actual_h < actual_a:
            match['result_type'] = 'Away Win'
        else:
            match['result_type'] = 'Draw'
    
    results_df = pd.DataFrame(predictions)
    for result_type in ['Home Win', 'Draw', 'Away Win']:
        subset = results_df[results_df['result_type'] == result_type]
        if len(subset) > 0:
            logger.info(f"{result_type:10s}: {len(subset):2d} matches | Exact: {subset['exact_match'].mean():.1%} | Outcome: {subset['correct_outcome'].mean():.1%}")
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE!")
    logger.info("="*80)
    
    return results_df


if __name__ == "__main__":
    test_recent_matches()
