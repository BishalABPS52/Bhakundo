"""Collect and process player availability data (injuries, suspensions, AFCON, etc.)"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.data.collectors.fpl_api import FPLDataCollector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def collect_current_availability():
    """Collect current player availability from FPL"""
    logger.info("="*70)
    logger.info("COLLECTING PLAYER AVAILABILITY DATA")
    logger.info("="*70)
    
    # Initialize FPL collector
    fpl = FPLDataCollector()
    
    # Get all player data
    logger.info("\n1. Fetching player data...")
    players_df = fpl.get_players_data()
    logger.info(f"   Retrieved {len(players_df)} players")
    
    # Get injuries and suspensions
    logger.info("\n2. Extracting availability issues...")
    availability_df = fpl.get_injuries_and_suspensions()
    logger.info(f"   Found {len(availability_df)} players with availability issues")
    
    # Get team mappings
    logger.info("\n3. Getting team mappings...")
    teams_df = fpl.get_teams_data()
    
    # Create team code to name mapping
    team_mapping = {
        'ARS': 'Arsenal FC', 'AVL': 'Aston Villa FC', 'BOU': 'AFC Bournemouth',
        'BRE': 'Brentford FC', 'BHA': 'Brighton & Hove Albion FC', 'BUR': 'Burnley FC',
        'CHE': 'Chelsea FC', 'CRY': 'Crystal Palace FC', 'EVE': 'Everton FC',
        'FUL': 'Fulham FC', 'LEE': 'Leeds United FC', 'LIV': 'Liverpool FC',
        'MCI': 'Manchester City FC', 'MUN': 'Manchester United FC', 'NEW': 'Newcastle United FC',
        'NFO': 'Nottingham Forest FC', 'SUN': 'Sunderland AFC', 'TOT': 'Tottenham Hotspur FC',
        'WHU': 'West Ham United FC', 'WOL': 'Wolverhampton Wanderers FC'
    }
    
    # Merge team information
    players_full = players_df.merge(teams_df[['fpl_team_id', 'short_name']], 
                                     left_on='team_code', right_on='fpl_team_id', how='left')
    
    # Map to full team names
    players_full['team_name'] = players_full['short_name'].map(team_mapping)
    
    # Save player data with availability info
    output_dir = Path('data/raw/player_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    players_full.to_csv(output_dir / 'fpl_players_current.csv', index=False)
    availability_df.to_csv(output_dir / 'player_availability_current.csv', index=False)
    
    logger.info(f"\n4. Saved player data to {output_dir}")
    
    # Display summary
    logger.info("\n" + "="*70)
    logger.info("AVAILABILITY SUMMARY")
    logger.info("="*70)
    
    status_counts = players_df['status'].value_counts()
    logger.info(f"\nPlayer Status Breakdown:")
    for status, count in status_counts.items():
        status_label = {
            'a': 'Available',
            'd': 'Doubtful',
            'i': 'Injured',
            's': 'Suspended',
            'u': 'Unavailable',
            'n': 'Not in squad'
        }.get(status, status)
        logger.info(f"  {status_label}: {count} players")
    
    # Show specific unavailable players
    unavailable = players_full[players_full['status'] != 'a'][
        ['player_name', 'team_name', 'position', 'status', 'news', 'chance_of_playing_next_round']
    ].copy()
    
    if not unavailable.empty:
        logger.info(f"\n{len(unavailable)} Players with Availability Issues:")
        logger.info("-"*70)
        for _, player in unavailable.head(20).iterrows():
            pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            status_label = {'d': 'Doubtful', 'i': 'Injured', 's': 'Suspended', 'u': 'Unavailable', 'n': 'Not in squad'}.get(player['status'], player['status'])
            chance = f"{player['chance_of_playing_next_round']}%" if pd.notna(player['chance_of_playing_next_round']) else "Unknown"
            logger.info(f"  • {player['player_name']:25s} ({player['team_name']:25s} - {pos_map.get(player['position'], 'N/A'):3s}) | {status_label:12s} | {chance:7s} | {player['news'][:50] if player['news'] else 'No details'}")
    
    return players_full, availability_df


def calculate_team_availability_impact(players_df):
    """Calculate team strength reduction due to unavailable key players"""
    logger.info("\n" + "="*70)
    logger.info("CALCULATING TEAM AVAILABILITY IMPACT")
    logger.info("="*70)
    
    # Define player importance weights based on position and stats
    players_df['importance_score'] = (
        players_df['total_points'] * 0.3 +
        players_df['form'] * 100 * 0.2 +
        players_df['influence'] * 0.2 +
        players_df['creativity'] * 0.15 +
        players_df['threat'] * 0.15
    )
    
    # Calculate availability percentage (0-100)
    players_df['availability_pct'] = players_df['chance_of_playing_next_round'].fillna(
        players_df['status'].map({'a': 100, 'd': 50, 'i': 0, 's': 0, 'u': 0, 'n': 0}).fillna(100)
    )
    
    # Group by team
    team_impact = []
    
    for team_name in players_df['team_name'].dropna().unique():
        team_players = players_df[players_df['team_name'] == team_name].copy()
        
        # Get top 11 players by importance
        top_11 = team_players.nlargest(11, 'importance_score')
        
        # Calculate available strength
        total_importance = top_11['importance_score'].sum()
        available_importance = (top_11['importance_score'] * top_11['availability_pct'] / 100).sum()
        
        availability_rate = (available_importance / total_importance * 100) if total_importance > 0 else 100
        
        # Count unavailable key players
        unavailable_count = len(top_11[top_11['availability_pct'] < 100])
        key_players_out = len(top_11[top_11['availability_pct'] == 0])
        
        team_impact.append({
            'team_name': team_name,
            'total_squad_size': len(team_players),
            'top_11_availability_rate': availability_rate,
            'players_with_issues': unavailable_count,
            'key_players_out': key_players_out,
            'squad_depth_score': len(team_players[team_players['total_points'] > 50]),
        })
    
    impact_df = pd.DataFrame(team_impact).sort_values('top_11_availability_rate')
    
    logger.info("\nTeam Availability Impact:")
    logger.info("-"*70)
    for _, team in impact_df.iterrows():
        logger.info(f"{team['team_name']:30s} | Availability: {team['top_11_availability_rate']:5.1f}% | "
                   f"Issues: {team['players_with_issues']:2.0f} | Out: {team['key_players_out']:2.0f} | "
                   f"Depth: {team['squad_depth_score']:2.0f}")
    
    # Save impact data
    impact_df.to_csv('data/raw/player_data/team_availability_impact.csv', index=False)
    logger.info(f"\nSaved team availability impact to data/raw/player_data/team_availability_impact.csv")
    
    return impact_df


def create_availability_features(team_name, players_df):
    """Create availability features for a specific team"""
    team_players = players_df[players_df['team_name'] == team_name].copy()
    
    if team_players.empty:
        return {
            'key_players_available_pct': 100,
            'squad_availability_rate': 100,
            'top_scorer_available': 1,
            'top_assister_available': 1,
            'key_defender_available': 1,
            'goalkeeper_available': 1,
            'unavailable_importance_loss': 0,
        }
    
    # Calculate importance scores
    team_players['importance_score'] = (
        team_players['total_points'] * 0.3 +
        team_players['form'] * 100 * 0.2 +
        team_players['influence'] * 0.2 +
        team_players['creativity'] * 0.15 +
        team_players['threat'] * 0.15
    )
    
    # Availability percentage
    team_players['availability_pct'] = team_players['chance_of_playing_next_round'].fillna(
        team_players['status'].map({'a': 100, 'd': 50, 'i': 0, 's': 0, 'u': 0, 'n': 0}).fillna(100)
    )
    
    # Get top players by position
    top_11 = team_players.nlargest(11, 'importance_score')
    top_scorer = team_players.nlargest(1, 'goals_scored')
    top_assister = team_players.nlargest(1, 'assists')
    top_defender = team_players[team_players['position'] == 2].nlargest(1, 'importance_score')
    goalkeeper = team_players[team_players['position'] == 1].nlargest(1, 'importance_score')
    
    # Calculate features
    total_importance = top_11['importance_score'].sum()
    available_importance = (top_11['importance_score'] * top_11['availability_pct'] / 100).sum()
    
    features = {
        'key_players_available_pct': (available_importance / total_importance * 100) if total_importance > 0 else 100,
        'squad_availability_rate': team_players['availability_pct'].mean(),
        'top_scorer_available': 1 if top_scorer['availability_pct'].iloc[0] >= 75 else 0 if not top_scorer.empty else 1,
        'top_assister_available': 1 if top_assister['availability_pct'].iloc[0] >= 75 else 0 if not top_assister.empty else 1,
        'key_defender_available': 1 if not top_defender.empty and top_defender['availability_pct'].iloc[0] >= 75 else 1 if top_defender.empty else 0,
        'goalkeeper_available': 1 if not goalkeeper.empty and goalkeeper['availability_pct'].iloc[0] >= 75 else 1 if goalkeeper.empty else 0,
        'unavailable_importance_loss': ((total_importance - available_importance) / total_importance * 100) if total_importance > 0 else 0,
    }
    
    return features


if __name__ == "__main__":
    # Collect current availability data
    players_df, availability_df = collect_current_availability()
    
    # Calculate team impact
    impact_df = calculate_team_availability_impact(players_df)
    
    # Example: Show features for a team
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE: Availability Features for Manchester City FC")
    logger.info("="*70)
    
    features = create_availability_features('Manchester City FC', players_df)
    for key, value in features.items():
        logger.info(f"  {key:35s}: {value:6.1f}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Player availability data collection complete!")
    logger.info("="*70)
