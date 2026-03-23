"""
Enhanced FastAPI Backend for Premier League Predictor 2025-26
Uses enhanced models with gameweek context and live standings
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio

# Import ML libraries (required for unpickling models)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from email.mime.multipart import MIMEMultipart
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import football API integration and ensemble predictor
from backend.football_api import football_api
from backend.ensemble_predictor import AlignmentEnsemble, format_prediction_with_confidence
from backend.database import Prediction, Actual, init_db, get_db_session
from backend.auth import verify_admin_credentials, verify_api_key, get_current_user

app = FastAPI(
    title="Premier League Predictor API 2025-26", 
    version="2.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Background task to update results periodically
async def periodic_result_updater():
    """Background task — runs every 30 min to sync finished match scores into the actual table."""
    while True:
        try:
            await asyncio.sleep(1800)  # Every 30 minutes
            print("🔄 Running automatic actual-table sync...")
            db = get_db_session()
            result = football_api.sync_actual_from_api(db)
            db.close()
            print(f"   ✓ Sync done — updated={result['updated']} inserted={result['inserted']} errors={result['errors']}")
        except Exception as e:
            print(f"   ⚠ Error in periodic updater: {e}")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup and start background tasks"""
    try:
        init_db()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"⚠ Database initialization warning: {e}")
        print("  Predictions will work without database, but history won't be saved")
    
    # Start background result updater
    asyncio.create_task(periodic_result_updater())
    print("✓ Started automatic result updater (runs every hour)")

# CORS middleware - Allow multiple frontend URLs with fallback support
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://bhakundo.vercel.app",
        "https://bhakundo-frontend.vercel.app",
        "https://*.vercel.app",  # All Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Premier League 2025-26 Teams
CURRENT_SEASON_TEAMS = [
    "Arsenal FC",
    "Aston Villa FC",
    "AFC Bournemouth",
    "Brentford FC",
    "Brighton & Hove Albion FC",
    "Burnley FC",  # Promoted
    "Chelsea FC",
    "Crystal Palace FC",
    "Everton FC",
    "Fulham FC",
    "Leeds United FC",  # Promoted
    "Liverpool FC",
    "Manchester City FC",
    "Manchester United FC",
    "Newcastle United FC",
    "Nottingham Forest FC",
    "Sunderland AFC",  # Promoted
    "Tottenham Hotspur FC",
    "West Ham United FC",
    "Wolverhampton Wanderers FC"
]

# Load enhanced models
try:
    # Determine base path (works for both local and Render deployment)
    base_path = Path(__file__).resolve().parent.parent
    models_dir = base_path / 'required' / 'data' / 'models'
    
    # Load score prediction model (supports both single and ensemble models)
    score_model_data = joblib.load(models_dir / 'score_prediction_model_enhanced_v3.pkl')
    
    # Check if ensemble or single model
    score_is_ensemble = 'home_models' in score_model_data
    
    if score_is_ensemble:
        # Ensemble model format
        score_model_home = score_model_data['home_models']
        score_model_away = score_model_data['away_models']
        score_home_weights = score_model_data['home_weights']
        score_away_weights = score_model_data['away_weights']
        print("✅ Loaded ensemble score model (XGB + LightGBM + Linear)")
    else:
        # Single / blended model format
        score_model_home = score_model_data['home_model']
        score_model_away = score_model_data['away_model']
        score_home_weights = None
        score_away_weights = None
        print("✅ Loaded blended score model (LGB + XGB v3)")
    
    score_scaler = score_model_data['scaler']
    score_features = score_model_data['feature_columns']
    score_max_goals = score_model_data.get('max_goals', 9)
    
    base_model_data = joblib.load(models_dir / 'base_outcome_model_enhanced.pkl')
    base_model = base_model_data['model']
    base_scaler = base_model_data['scaler']
    base_features = base_model_data['feature_columns']
    base_label_mapping = base_model_data.get('reverse_mapping', {0: 'A', 1: 'D', 2: 'H'})
    
    lineup_model_data = joblib.load(models_dir / 'lineup_model_enhanced.pkl')
    lineup_model = lineup_model_data['model']
    lineup_scaler = lineup_model_data['scaler']
    lineup_features = lineup_model_data['feature_columns']
    lineup_label_mapping = lineup_model_data.get('reverse_mapping', {0: 'A', 1: 'D', 2: 'H'})
    
    print("✅ Enhanced models v3 loaded successfully")
    print(f"📊 Features: {len(score_features)} (Base/Lineup/Score)")
    print(f"⚽ Score range: 0-{score_max_goals} goals")
    
    # Initialize ensemble predictor
    ensemble = AlignmentEnsemble()
    print("🤝 Alignment ensemble initialized")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Storage for saved predictions
PREDICTIONS_FILE = 'data/saved_predictions.json'

def load_saved_predictions():
    """Load saved predictions from file"""
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_prediction_to_file(match_id: str, prediction_data: dict):
    """Save a prediction to file"""
    predictions = load_saved_predictions()
    predictions[str(match_id)] = prediction_data
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)

def get_saved_prediction(match_id: str):
    """Get a saved prediction for a match"""
    predictions = load_saved_predictions()
    return predictions.get(str(match_id))

# Load match data and enhanced features
matches_df = pd.read_csv(base_path / 'required' / 'data' / 'raw' / 'pl' / 'pl_all_seasons_combined.csv')
matches_df['match_date'] = pd.to_datetime(matches_df['match_date'], errors='coerce')
matches_df = matches_df[matches_df['status'].str.upper().isin(['FINISHED', 'COMPLETE'])].copy()

# Drop rows with NaN goals
matches_df = matches_df.dropna(subset=['home_goals', 'away_goals']).copy()

# Convert goals to integers
matches_df['home_goals'] = matches_df['home_goals'].astype(int)
matches_df['away_goals'] = matches_df['away_goals'].astype(int)

# Load player availability
try:
    availability_df = pd.read_csv(base_path / 'required' / 'data' / 'raw' / 'pl' / 'player_data' / 'team_availability_impact.csv')
    if 'team_name' in availability_df.columns:
        availability_df = availability_df.rename(columns={'team_name': 'team'})
    if 'top_11_availability_rate' in availability_df.columns:
        availability_df['availability_rate'] = availability_df['top_11_availability_rate']
    if 'squad_depth_score' in availability_df.columns:
        availability_df['squad_depth'] = availability_df['squad_depth_score']
except:
    # Default availability
    availability_df = pd.DataFrame({
        'team': CURRENT_SEASON_TEAMS,
        'availability_rate': [0.85] * len(CURRENT_SEASON_TEAMS),
        'key_players_out': [0] * len(CURRENT_SEASON_TEAMS),
        'squad_depth': [0.7] * len(CURRENT_SEASON_TEAMS)
    })


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    gameweek: Optional[int] = None
    home_formation: Optional[str] = None
    away_formation: Optional[str] = None
    match_id: Optional[str] = None  # Match ID from football API


class StandingsEntry(BaseModel):
    position: int
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    form: List[str]  # Last 5 results: W, D, L


def calculate_current_standings() -> List[StandingsEntry]:
    """Get current standings with live API first, then local historical fallback."""
    # Prefer live API standings so table updates automatically without server restarts.
    try:
        api_rows = football_api.get_standings()
        if api_rows:
            result = []
            for row in api_rows:
                form_raw = row.get('form', [])
                if isinstance(form_raw, str):
                    form = [x.strip() for x in form_raw.split(',') if x.strip() in {'W', 'D', 'L'}]
                elif isinstance(form_raw, list):
                    form = [str(x).strip() for x in form_raw if str(x).strip() in {'W', 'D', 'L'}]
                else:
                    form = []

                result.append(StandingsEntry(
                    position=int(row.get('position') or 0),
                    team=row.get('team', ''),
                    played=int(row.get('played') or 0),
                    won=int(row.get('won') or 0),
                    drawn=int(row.get('drawn') or 0),
                    lost=int(row.get('lost') or 0),
                    goals_for=int(row.get('gf', row.get('goals_for', 0)) or 0),
                    goals_against=int(row.get('ga', row.get('goals_against', 0)) or 0),
                    goal_difference=int(row.get('gd', row.get('goal_difference', 0)) or 0),
                    points=int(row.get('points') or 0),
                    form=form[-5:]
                ))

            if result:
                return sorted(result, key=lambda x: x.position)
    except Exception as e:
        print(f"⚠️ Live standings fallback triggered: {e}")

    # Fallback: calculate from preloaded historical dataframe.
    # Get all finished matches from current season
    current_season = matches_df[matches_df['season'] == '2025-26'].copy()
    
    standings = {}
    
    for team in CURRENT_SEASON_TEAMS:
        team_matches = current_season[
            (current_season['home_team'] == team) | (current_season['away_team'] == team)
        ].sort_values('match_date')
        
        stats = {
            'team': team,
            'played': len(team_matches),
            'won': 0,
            'drawn': 0,
            'lost': 0,
            'goals_for': 0,
            'goals_against': 0,
            'points': 0,
            'form': []
        }
        
        for _, match in team_matches.iterrows():
            # Use home_goals/away_goals if available, fallback to home_goals/away_goals
            home_goals = match.get('home_goals') if pd.notna(match.get('home_goals')) else match.get('home_goals', 0)
            away_goals = match.get('away_goals') if pd.notna(match.get('away_goals')) else match.get('away_goals', 0)
            
            # Skip if still NaN
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue
                
            if match['home_team'] == team:
                gf, ga = int(home_goals), int(away_goals)
            else:
                gf, ga = int(away_goals), int(home_goals)
            
            stats['goals_for'] += gf
            stats['goals_against'] += ga
            
            if gf > ga:
                stats['won'] += 1
                stats['points'] += 3
                stats['form'].append('W')
            elif gf == ga:
                stats['drawn'] += 1
                stats['points'] += 1
                stats['form'].append('D')
            else:
                stats['lost'] += 1
                stats['form'].append('L')
        
        stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
        stats['form'] = stats['form'][-5:]  # Last 5 matches
        standings[team] = stats
    
    # Sort by points, then GD, then GF
    sorted_standings = sorted(
        standings.values(),
        key=lambda x: (x['points'], x['goal_difference'], x['goals_for']),
        reverse=True
    )
    
    # Add positions
    result = []
    for i, entry in enumerate(sorted_standings, 1):
        result.append(StandingsEntry(
            position=i,
            **entry
        ))
    
    return result


def fill_missing_score_features(features: dict) -> dict:
    """
    Map v3 feature names → score-model feature names and fill in any gaps.
    Call this before running the score model so all score_features columns exist.
    """
    f = features.copy()
    LEAGUE_AVG = 1.5

    # ── ELO (estimated from standings position) ─────────────────────────────
    home_pos = f.get('home_position', 10)
    away_pos = f.get('away_position', 10)
    h_elo = 1500 - (home_pos - 1) * 28
    a_elo = 1500 - (away_pos - 1) * 28
    f.setdefault('home_elo', h_elo)
    f.setdefault('away_elo', a_elo)
    f.setdefault('elo_diff', h_elo - a_elo)
    f.setdefault('elo_ratio', h_elo / max(a_elo, 1))
    h_elo_win_prob = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
    f.setdefault('home_elo_win_prob', h_elo_win_prob)
    f.setdefault('away_elo_win_prob', 1 - h_elo_win_prob)
    f.setdefault('elo_momentum_home', f['home_elo'] * f.get('home_form_points_l5', 7) / 15.0)

    # ── Attack / defense strength ────────────────────────────────────────────
    h_goals     = f.get('home_venue_goals',     LEAGUE_AVG)
    h_conceded  = f.get('home_venue_conceded',  LEAGUE_AVG)
    a_goals     = f.get('away_venue_goals',     LEAGUE_AVG)
    a_conceded  = f.get('away_venue_conceded',  LEAGUE_AVG)
    f.setdefault('home_attack_strength',  h_goals    / LEAGUE_AVG)
    f.setdefault('home_defense_strength', h_conceded / LEAGUE_AVG)
    f.setdefault('away_attack_strength',  a_goals    / LEAGUE_AVG)
    f.setdefault('away_defense_strength', a_conceded / LEAGUE_AVG)
    f.setdefault('home_avg_goals_scored',    h_goals)
    f.setdefault('home_avg_goals_conceded',  h_conceded)
    f.setdefault('away_avg_goals_scored',    a_goals)
    f.setdefault('away_avg_goals_conceded',  a_conceded)
    f.setdefault('home_goal_difference',  f.get('home_gd', 0) / 10.0)
    f.setdefault('away_goal_difference',  f.get('away_gd', 0) / 10.0)
    f.setdefault('home_win_rate',       f.get('home_form_win_rate_l5', 0.33))
    f.setdefault('away_win_rate',       f.get('away_form_win_rate_l5', 0.33))
    f.setdefault('home_points_per_game', f.get('home_venue_ppg', 1.5))
    f.setdefault('away_points_per_game', f.get('away_venue_ppg', 1.0))

    # ── Last-5 overall ───────────────────────────────────────────────────────
    def _last5(pref):
        pts   = f.get(f'{pref}_form_points_l5', 7)
        goals = f.get(f'{pref}_form_goals_l5',    1.5) * 5
        conc  = f.get(f'{pref}_form_conceded_l5', 1.5) * 5
        wins  = f.get(f'{pref}_form_wins_l5',     2)
        draws = max(0, int(pts - wins * 3))
        losses= max(0, 5 - wins - draws)
        cs    = f.get(f'{pref}_form_clean_sheets_l5', 1)
        f.setdefault(f'{pref}_last5_overall_points',         pts)
        f.setdefault(f'{pref}_last5_overall_goals_scored',   goals)
        f.setdefault(f'{pref}_last5_overall_goals_conceded', conc)
        f.setdefault(f'{pref}_last5_overall_wins',           wins)
        f.setdefault(f'{pref}_last5_overall_draws',          draws)
        f.setdefault(f'{pref}_last5_overall_losses',         losses)
        f.setdefault(f'{pref}_last5_overall_clean_sheets',   cs)
    _last5('home')
    _last5('away')

    # ── Last-5 venue-specific ────────────────────────────────────────────────
    h_vwr  = f.get('home_venue_win_rate', 0.4)
    h_vcs  = f.get('home_venue_clean_sheets', 0.25)
    hw     = int(h_vwr * 5)
    hd     = max(0, 5 - hw - 2)
    hl     = max(0, 5 - hw - hd)
    f.setdefault('home_last5_home_points',          f.get('home_venue_ppg', 1.5) * 5)
    f.setdefault('home_last5_home_goals_scored',    h_goals * 5)
    f.setdefault('home_last5_home_goals_conceded',  h_conceded * 5)
    f.setdefault('home_last5_home_wins',            hw)
    f.setdefault('home_last5_home_draws',           hd)
    f.setdefault('home_last5_home_losses',          hl)
    f.setdefault('home_last5_home_clean_sheets',    int(h_vcs * 5))

    a_vwr  = f.get('away_venue_win_rate', 0.25)
    a_vcs  = f.get('away_venue_clean_sheets', 0.15)
    aw     = int(a_vwr * 5)
    ad     = max(0, 5 - aw - 2)
    al     = max(0, 5 - aw - ad)
    f.setdefault('away_last5_away_points',          f.get('away_venue_ppg', 1.0) * 5)
    f.setdefault('away_last5_away_goals_scored',    a_goals * 5)
    f.setdefault('away_last5_away_goals_conceded',  a_conceded * 5)
    f.setdefault('away_last5_away_wins',            aw)
    f.setdefault('away_last5_away_draws',           ad)
    f.setdefault('away_last5_away_losses',          al)
    f.setdefault('away_last5_away_clean_sheets',    int(a_vcs * 5))

    # ── Derived matchup features ─────────────────────────────────────────────
    f.setdefault('home_attack_vs_away_defense',
                 f['home_attack_strength'] / max(f['away_defense_strength'], 0.1))
    f.setdefault('away_attack_vs_home_defense',
                 f['away_attack_strength'] / max(f['home_defense_strength'], 0.1))
    f.setdefault('home_xg_estimate', f['home_attack_strength'] * LEAGUE_AVG)
    f.setdefault('away_xg_estimate', f['away_attack_strength'] * LEAGUE_AVG)
    f.setdefault('home_goal_diff_last5',
                 f.get('home_form_goals_l5', LEAGUE_AVG) - f.get('home_form_conceded_l5', LEAGUE_AVG))
    f.setdefault('away_goal_diff_last5',
                 f.get('away_form_goals_l5', LEAGUE_AVG) - f.get('away_form_conceded_l5', LEAGUE_AVG))
    f.setdefault('home_win_streak', min(f.get('home_form_wins_l5', 2), 5))
    f.setdefault('away_win_streak', min(f.get('away_form_wins_l5', 1), 5))
    f.setdefault('home_unbeaten_last5',
                 min(f.get('home_form_wins_l5', 2) + f.get('home_last5_overall_draws', 1), 5))
    f.setdefault('away_unbeaten_last5',
                 min(f.get('away_form_wins_l5', 2) + f.get('away_last5_overall_draws', 1), 5))
    f.setdefault('home_scoring_form',
                 f.get('home_form_goals_l5', LEAGUE_AVG) / max(f.get('away_form_conceded_l5', LEAGUE_AVG), 0.1))
    f.setdefault('away_scoring_form',
                 f.get('away_form_goals_l5', LEAGUE_AVG) / max(f.get('home_form_conceded_l5', LEAGUE_AVG), 0.1))
    f.setdefault('home_defensive_form', 1 / max(f.get('home_form_conceded_l5', LEAGUE_AVG), 0.1))
    f.setdefault('away_defensive_form', 1 / max(f.get('away_form_conceded_l5', LEAGUE_AVG), 0.1))

    # ── Venue stats ──────────────────────────────────────────────────────────
    f.setdefault('home_home_win_rate',    f.get('home_venue_win_rate',  0.4))
    f.setdefault('home_home_goals_avg',   h_goals)
    f.setdefault('home_home_conceded_avg', h_conceded)
    f.setdefault('away_away_loss_rate',   1 - f.get('away_venue_win_rate', 0.25))
    f.setdefault('away_away_goals_avg',   a_goals)
    f.setdefault('away_away_conceded_avg', a_conceded)
    f.setdefault('venue_advantage',
                 f['home_home_win_rate'] - (1 - f['away_away_loss_rate']) + 0.1)
    f.setdefault('venue_goal_diff', h_goals - a_conceded)

    # ── Rolling PPG & differentials ──────────────────────────────────────────
    h_ppg = f.get('home_form_points_l5', 7) / 5
    a_ppg = f.get('away_form_points_l5', 5) / 5
    f.setdefault('home_rolling_ppg',           h_ppg)
    f.setdefault('away_rolling_ppg',           a_ppg)
    f.setdefault('home_rolling_goals_for',     f.get('home_form_goals_l5',    LEAGUE_AVG))
    f.setdefault('away_rolling_goals_for',     f.get('away_form_goals_l5',    LEAGUE_AVG))
    f.setdefault('home_rolling_goals_against', f.get('home_form_conceded_l5', LEAGUE_AVG))
    f.setdefault('away_rolling_goals_against', f.get('away_form_conceded_l5', LEAGUE_AVG))
    f.setdefault('ppg_diff',           h_ppg - a_ppg)
    f.setdefault('goals_for_diff',
                 f['home_rolling_goals_for']  - f['away_rolling_goals_for'])
    f.setdefault('goals_against_diff',
                 f['home_rolling_goals_against'] - f['away_rolling_goals_against'])
    f.setdefault('overall_strength_diff',
                 f['home_attack_strength'] - f['away_defense_strength'])
    f.setdefault('form_diff',
                 f.get('home_form_points_l5', 7) - f.get('away_form_points_l5', 5))
    f.setdefault('momentum_score', (h_ppg - a_ppg) * 0.5)

    # ── Clean-sheet rates ────────────────────────────────────────────────────
    f.setdefault('home_clean_sheets_rate',  f.get('home_venue_clean_sheets', 0.25))
    f.setdefault('away_clean_sheets_rate',  f.get('away_venue_clean_sheets', 0.15))
    f.setdefault('home_clean_sheets_home',  f['home_last5_home_clean_sheets'] / 5)
    f.setdefault('away_clean_sheets_away',  f['away_last5_away_clean_sheets'] / 5)
    f.setdefault('home_defensive_solidity', f['home_clean_sheets_rate'])
    f.setdefault('away_defensive_solidity', f['away_clean_sheets_rate'])

    # ── Ratio / composite features ───────────────────────────────────────────
    f.setdefault('total_expected_goals',
                 f['home_xg_estimate'] + f['away_xg_estimate'])
    f.setdefault('expected_goal_ratio',
                 f['home_xg_estimate'] / max(f['away_xg_estimate'], 0.1))
    f.setdefault('strength_ratio',
                 f['home_attack_strength'] / max(f['away_defense_strength'], 0.1))
    f.setdefault('defense_ratio',
                 f['home_defense_strength'] / max(f['away_attack_strength'], 0.1))
    f.setdefault('win_rate_diff', f['home_win_rate'] - f['away_win_rate'])
    q_home = f['home_attack_strength'] + (1 - f['home_defense_strength'])
    q_away = f['away_attack_strength'] + (1 - f['away_defense_strength'])
    f.setdefault('quality_differential',         q_home - q_away)
    f.setdefault('reverse_quality_differential', q_away - q_home)
    f.setdefault('expected_home_xg_strength',
                 f['home_xg_estimate'] * f['home_attack_strength'])
    f.setdefault('expected_away_xg_strength',
                 f['away_xg_estimate'] * f['away_attack_strength'])

    # ── Rest / days ──────────────────────────────────────────────────────────
    f.setdefault('home_days_since_last_match', 7)
    f.setdefault('away_days_since_last_match', 7)
    f.setdefault('rest_differential', 0)

    return f


def calculate_team_features(team: str, opponent: str, is_home: bool, gameweek: int = None) -> Dict:
    """Calculate comprehensive team features for prediction"""
    
def get_team_form(team: str, up_to_gameweek: int = None) -> List[str]:
    """Get last 5 matches form for a team (W/D/L)"""
    current_season = matches_df[matches_df['season'] == '2025-26'].copy()
    
    # Filter matches up to the specified gameweek if provided
    if up_to_gameweek:
        current_season = current_season[current_season['gameweek'] < up_to_gameweek]
    
    team_matches = current_season[
        (current_season['home_team'] == team) | (current_season['away_team'] == team)
    ].sort_values('match_date')
    
    form = []
    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            gf, ga = match['home_goals'], match['away_goals']
        else:
            gf, ga = match['away_goals'], match['home_goals']
        
        if gf > ga:
            form.append('W')
        elif gf == ga:
            form.append('D')
        else:
            form.append('L')
    
    return form[-5:]  # Last 5 matches


def calculate_team_features(team: str, opponent: str, is_home: bool, gameweek: int = None) -> Dict:
    """Calculate 125 features matching training script (v2+v3 combined)"""
    team_matches = matches_df[
        (matches_df['home_team'] == team) | (matches_df['away_team'] == team)
    ].sort_values('match_date').tail(20)
    
    if len(team_matches) < 5:
        return None
    
    features = {}
    prefix = 'home' if is_home else 'away'
    gw = gameweek if gameweek else 20
    
    # Match context
    if is_home:
        features['gameweek'] = gw
        features['season_progress'] = gw / 38.0
    
    # Get standings
    standings = calculate_current_standings()
    team_standing = next((s for s in standings if s.team == team), None)
    position = team_standing.position if team_standing else 10
    points = team_standing.points if team_standing else 0
    gd = team_standing.goal_difference if team_standing else 0
    gf = team_standing.goals_for if team_standing else 0
    ga = team_standing.goals_against if team_standing else 0
    
    # Form calculation functions
    def calc_form(matches_list, n):
        recent = matches_list.tail(min(n, len(matches_list)))
        pts, gs_list, gc_list, wins, clean_sheets = 0, [], [], 0, 0
        for _, m in recent.iterrows():
            if m['home_team'] == team:
                gs, gc = m['home_goals'], m['away_goals']
            else:
                gs, gc = m['away_goals'], m['home_goals']
            gs_list.append(gs)
            gc_list.append(gc)
            if gs > gc: pts += 3; wins += 1
            elif gs == gc: pts += 1
            if gc == 0: clean_sheets += 1
        return {
            'points': pts, 'goals': np.mean(gs_list) if gs_list else 1.5,
            'conceded': np.mean(gc_list) if gc_list else 1.5,
            'wins': wins, 'win_rate': wins/len(recent) if len(recent) > 0 else 0.33,
            'clean_sheets': clean_sheets
        }
    
    # Venue-specific stats
    if is_home:
        venue_matches = matches_df[matches_df['home_team'] == team].tail(10)
    else:
        venue_matches = matches_df[matches_df['away_team'] == team].tail(10)
    
    venue_pts, venue_gs, venue_gc, venue_wins, venue_cs = 0, [], [], 0, 0
    for _, m in venue_matches.iterrows():
        if is_home:
            gs, gc = m['home_goals'], m['away_goals']
        else:
            gs, gc = m['away_goals'], m['home_goals']
        venue_gs.append(gs)
        venue_gc.append(gc)
        if gs > gc: venue_pts += 3; venue_wins += 1
        elif gs == gc: venue_pts += 1
        if gc == 0: venue_cs += 1
    
    form_l5 = calc_form(team_matches, 5)
    form_l10 = calc_form(team_matches, 10)
    form_l20 = calc_form(team_matches, 20)
    
    # V3 Form features (L5)
    features[f'{prefix}_form_points_l5'] = form_l5['points']
    features[f'{prefix}_form_goals_l5'] = form_l5['goals']
    features[f'{prefix}_form_conceded_l5'] = form_l5['conceded']
    features[f'{prefix}_form_wins_l5'] = form_l5['wins']
    features[f'{prefix}_form_win_rate_l5'] = form_l5['win_rate']
    features[f'{prefix}_form_clean_sheets_l5'] = form_l5['clean_sheets']
    
    # V3 Form features (L10)
    features[f'{prefix}_form_points_l10'] = form_l10['points']
    features[f'{prefix}_form_goals_l10'] = form_l10['goals']
    features[f'{prefix}_form_conceded_l10'] = form_l10['conceded']
    features[f'{prefix}_form_win_rate_l10'] = form_l10['win_rate']
    
    # V3 Venue stats
    features[f'{prefix}_venue_ppg'] = venue_pts / len(venue_matches) if len(venue_matches) > 0 else 1.5
    features[f'{prefix}_venue_goals'] = np.mean(venue_gs) if venue_gs else 1.5
    features[f'{prefix}_venue_conceded'] = np.mean(venue_gc) if venue_gc else 1.5
    features[f'{prefix}_venue_win_rate'] = venue_wins / len(venue_matches) if len(venue_matches) > 0 else 0.33
    features[f'{prefix}_venue_clean_sheets'] = venue_cs / len(venue_matches) if len(venue_matches) > 0 else 0.2
    
    # V3 Standings
    features[f'{prefix}_position'] = position
    features[f'{prefix}_points'] = points
    features[f'{prefix}_gd'] = gd
    features[f'{prefix}_gf'] = gf
    features[f'{prefix}_ga'] = ga
    
    # V2 Additional form metrics
    features[f'{prefix}_ppg_l5'] = form_l5['points'] / 5
    features[f'{prefix}_ppg_l10'] = form_l10['points'] / 10
    features[f'{prefix}_ppg_l20'] = form_l20['points'] / 20
    features[f'{prefix}_avg_goals_scored_l5'] = form_l5['goals']
    features[f'{prefix}_avg_goals_scored_l10'] = form_l10['goals']
    features[f'{prefix}_avg_goals_scored_l20'] = form_l20['goals']
    features[f'{prefix}_avg_goals_conceded_l5'] = form_l5['conceded']
    features[f'{prefix}_avg_goals_conceded_l10'] = form_l10['conceded']
    features[f'{prefix}_avg_goals_conceded_l20'] = form_l20['conceded']
    features[f'{prefix}_win_rate_l5'] = form_l5['win_rate']
    features[f'{prefix}_win_rate_l10'] = form_l10['win_rate']
    features[f'{prefix}_win_rate_l20'] = form_l20['win_rate']
    features[f'{prefix}_momentum'] = form_l5['points'] - form_l10['points'] / 2
    
    # V2 Recent gameweek stats
    recent_10 = team_matches.tail(10)
    gw_pts, gw_gs, gw_gc, gw_wins, gw_draws, gw_losses = 0, [], [], 0, 0, 0
    for _, m in recent_10.iterrows():
        if m['home_team'] == team:
            gs, gc = m['home_goals'], m['away_goals']
        else:
            gs, gc = m['away_goals'], m['home_goals']
        gw_gs.append(gs)
        gw_gc.append(gc)
        if gs > gc: gw_pts += 3; gw_wins += 1
        elif gs == gc: gw_pts += 1; gw_draws += 1
        else: gw_losses += 1
    
    features[f'{prefix}_recent_gw_ppg'] = gw_pts / len(recent_10) if len(recent_10) > 0 else 1.5
    features[f'{prefix}_recent_gw_goals'] = np.mean(gw_gs) if gw_gs else 1.5
    features[f'{prefix}_recent_gw_conceded'] = np.mean(gw_gc) if gw_gc else 1.5
    features[f'{prefix}_recent_gw_wins'] = gw_wins
    features[f'{prefix}_recent_gw_draws'] = gw_draws
    features[f'{prefix}_recent_gw_losses'] = gw_losses
    features[f'{prefix}_recent_gw_win_rate'] = gw_wins / len(recent_10) if len(recent_10) > 0 else 0.33
    
    # V2 Position & previous GW
    features[f'{prefix}_position_strength'] = 1.0 if position <= 6 else 0.5 if position <= 14 else 0.0
    if len(team_matches) > 0:
        last = team_matches.iloc[-1]
        if last['home_team'] == team:
            prev_gs, prev_gc = last['home_goals'], last['away_goals']
        else:
            prev_gs, prev_gc = last['away_goals'], last['home_goals']
        features[f'{prefix}_prev_gw_result'] = 3 if prev_gs > prev_gc else 1 if prev_gs == prev_gc else 0
        features[f'{prefix}_prev_gw_goals'] = prev_gs
        features[f'{prefix}_prev_gw_conceded'] = prev_gc
    else:
        features[f'{prefix}_prev_gw_result'] = 1
        features[f'{prefix}_prev_gw_goals'] = 1
        features[f'{prefix}_prev_gw_conceded'] = 1
    
    # V2 Player availability (defaults for now)
    features[f'{prefix}_availability_rate'] = 0.85
    features[f'{prefix}_key_players_out'] = 1
    features[f'{prefix}_squad_depth'] = 0.7
    features[f'{prefix}_goalkeeper_available'] = 1
    features[f'{prefix}_defenders_available'] = 4
    features[f'{prefix}_midfielders_available'] = 4
    features[f'{prefix}_top_scorer_available'] = 1
    features[f'{prefix}_days_since_last_match'] = 7
    
    return features


# Protected documentation endpoints (admin only)
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

@app.get("/docs", include_in_schema=False)
async def get_documentation(username: str = Depends(verify_admin_credentials)):
    """Protected Swagger UI documentation - requires admin login"""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")

@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation(username: str = Depends(verify_admin_credentials)):
    """Protected ReDoc documentation - requires admin login"""
    return get_redoc_html(openapi_url="/openapi.json", title="API Docs")

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint(username: str = Depends(verify_admin_credentials)):
    """Protected OpenAPI schema - requires admin login"""
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


# Public root endpoint (no auth required)
@app.get("/")
async def root():
    return {
        "message": "Bhakundo - Prepare, Predict & Play API",
        "version": "3.0.0",
        "models": "Trained on Reliable datas with three models - Base, Lineup, Score and aligned using ensemble model.",
        "season": "2025-26",
        "note": "API key required for endpoints. Contact admin for access and visit bishalshrestha.com.np for more."
    }

# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and frontend connectivity testing"""
    return {
        "status": "healthy",
        "service": "bhakundo-predictor-api",
        "timestamp": datetime.now().isoformat()
    }

[{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1005",
	"severity": 8,
	"message": "',' expected.",
	"source": "ts",
	"startLineNumber": 88,
	"startColumn": 20,
	"endLineNumber": 88,
	"endColumn": 25,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "8010",
	"severity": 8,
	"message": "Type annotations can only be used in TypeScript files.",
	"source": "ts",
	"startLineNumber": 89,
	"startColumn": 9,
	"endLineNumber": 89,
	"endColumn": 11,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1005",
	"severity": 8,
	"message": "',' expected.",
	"source": "ts",
	"startLineNumber": 89,
	"startColumn": 12,
	"endLineNumber": 89,
	"endColumn": 13,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1005",
	"severity": 8,
	"message": "';' expected.",
	"source": "ts",
	"startLineNumber": 89,
	"startColumn": 36,
	"endLineNumber": 89,
	"endColumn": 47,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1128",
	"severity": 8,
	"message": "Declaration or statement expected.",
	"source": "ts",
	"startLineNumber": 90,
	"startColumn": 9,
	"endLineNumber": 90,
	"endColumn": 13,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1005",
	"severity": 8,
	"message": "',' expected.",
	"source": "ts",
	"startLineNumber": 91,
	"startColumn": 9,
	"endLineNumber": 91,
	"endColumn": 14,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1128",
	"severity": 8,
	"message": "Declaration or statement expected.",
	"source": "ts",
	"startLineNumber": 96,
	"startColumn": 3,
	"endLineNumber": 96,
	"endColumn": 4,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1128",
	"severity": 8,
	"message": "Declaration or statement expected.",
	"source": "ts",
	"startLineNumber": 96,
	"startColumn": 4,
	"endLineNumber": 96,
	"endColumn": 5,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1005",
	"severity": 8,
	"message": "';' expected.",
	"source": "ts",
	"startLineNumber": 96,
	"startColumn": 8,
	"endLineNumber": 96,
	"endColumn": 9,
	"origin": "extHost1"
},{
	"resource": "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/frontend/components/PredictorPL.jsx",
	"owner": "typescript",
	"code": "1128",
	"severity": 8,
	"message": "Declaration or statement expected.",
	"source": "ts",
	"startLineNumber": 1125,
	"startColumn": 1,
	"endLineNumber": 1125,
	"endColumn": 2,
	"origin": "extHost1"
}]
@app.get("/config")
async def get_config(user: str = Depends(get_current_user)):
    """Get API configuration including current gameweek"""
    try:
        return {
            "current_gameweek": football_api.get_current_gameweek(),
            "season": "2025-26",
            "total_gameweeks": 38
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/gameweek")
async def set_current_gameweek(gameweek: int, user: str = Depends(get_current_user)):
    """Set current gameweek dynamically from frontend"""
    try:
        if gameweek < 1 or gameweek > 38:
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        # Store in config file
        config_path = Path(__file__).parent.parent / 'required' / 'data' / 'config.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump({'current_gameweek': gameweek}, f)
        
        return {
            "success": True,
            "current_gameweek": gameweek,
            "message": f"Gameweek updated to {gameweek}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Protected endpoints (require API key from frontend OR admin credentials)
@app.get("/teams")
async def get_teams(user: str = Depends(get_current_user)):
    """Get current season teams"""
    return {"teams": sorted(CURRENT_SEASON_TEAMS)}


@app.get("/smart-gameweek")
async def get_smart_gameweek(user: str = Depends(get_current_user)):
    """
    Auto-detect the last completed GW and upcoming GW.
    Priority order:
      1. Live football-data.org API (currentMatchday)
      2. DB Actual table (max FINISHED gameweek)
      3. config.json fallback
    Also syncs config.json so the rest of the backend stays aligned.
    """
    try:
        from sqlalchemy import func as sa_func
        import json as _json

        # 1. Try live API first — most reliable source of currentMatchday
        season_info = football_api.get_season_info()
        api_current_matchday = season_info.get('current_matchday')  # active/upcoming GW
        api_source = season_info.get('source', 'config')

        # 2. DB: highest GW with FINISHED matches = last completed GW
        db = get_db_session()
        try:
            db_last_completed = (
                db.query(sa_func.max(Actual.gameweek))
                .filter(Actual.status == 'FINISHED')
                .scalar()
            )
        finally:
            db.close()

        # Determine last_completed and upcoming:
        # - If API says currentMatchday is X, then last_completed = X-1 (if X>1) or X
        # - DB is ground-truth for what's actually finished
        if db_last_completed:
            last_completed = int(db_last_completed)
        elif api_current_matchday and api_source == 'api':
            last_completed = max(int(api_current_matchday) - 1, 1)
        else:
            last_completed = football_api.get_current_gameweek() - 1

        # upcoming: prefer live API currentMatchday (it advances when a new GW starts)
        if api_current_matchday and api_source == 'api':
            upcoming = int(api_current_matchday)
        else:
            upcoming = min(last_completed + 1, 38)

        # Sync config.json so backend routing stays in sync
        try:
            config_path = Path(__file__).parent.parent / 'required' / 'data' / 'config.json'
            with open(config_path, 'w') as f:
                _json.dump({"current_gameweek": upcoming}, f)
        except Exception:
            pass

        return {
            "last_completed_gameweek": last_completed,
            "upcoming_gameweek": upcoming,
            "results_gameweek": last_completed,
            "fixtures_gameweek": upcoming,
            "source": api_source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/standings")
async def get_standings(user: str = Depends(get_current_user)):
    """Get current Premier League standings"""
    try:
        standings = calculate_current_standings()
        return {"standings": standings, "last_updated": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fixtures")
async def get_fixtures(gameweek: Optional[int] = None, user: str = Depends(get_current_user)):
    """Get fixtures for a specific gameweek - shows upcoming and ongoing matches"""
    try:
        current_gw = football_api.get_current_gameweek()
        
        if not gameweek:
            gameweek = current_gw
        
        # Get matches from football API
        matches = football_api.get_matches(gameweek=gameweek)
        
        # Fallback to DB (Actual table) if live API returned nothing
        if not matches:
            db_sess = get_db_session()
            try:
                actuals = db_sess.query(Actual).filter(Actual.gameweek == gameweek).all()
                matches = [{
                    'id': a.match_id,
                    'match_id': a.match_id,
                    'gameweek': a.gameweek,
                    'date': a.updated_at.isoformat() if a.updated_at else None,
                    'home_team': a.home_team,
                    'away_team': a.away_team,
                    'home_score': a.actual_home_goals,
                    'away_score': a.actual_away_goals,
                    'status': a.status or ('FINISHED' if a.actual_home_goals is not None else 'SCHEDULED'),
                    'venue': football_api.stadiums.get(a.home_team, 'Stadium')
                } for a in actuals]
            finally:
                db_sess.close()

        # Second fallback: Prediction table — covers upcoming GWs where user already predicted
        # (Actual table empty for future GWs; Prediction table has them after user clicks Predict)
        if not matches:
            db_sess = get_db_session()
            try:
                preds = db_sess.query(Prediction).filter(Prediction.gameweek == gameweek).all()
                seen_mid: dict = {}
                for p in preds:
                    if p.match_id and p.match_id not in seen_mid:
                        seen_mid[p.match_id] = p
                matches = [{
                    'id': p.match_id,
                    'match_id': p.match_id,
                    'gameweek': p.gameweek,
                    'date': p.created_at.isoformat() if p.created_at else None,
                    'home_team': p.home_team,
                    'away_team': p.away_team,
                    'home_score': None,
                    'away_score': None,
                    'status': 'SCHEDULED',
                    'venue': football_api.stadiums.get(p.home_team, 'Stadium')
                } for p in seen_mid.values()]
            finally:
                db_sess.close()

        # Ensure each match has match_id field
        for m in matches:
            if 'match_id' not in m:
                m['match_id'] = str(m.get('id', ''))

        # Resolve canonical match_id: replace FPL/API IDs with the Prediction/Actual
        # table's own match_id so frontend predictions align correctly.
        _db_canon = get_db_session()
        try:
            for m in matches:
                home = m.get('home_team', '')
                away  = m.get('away_team', '')
                gw    = m.get('gameweek', gameweek)
                act = _db_canon.query(Actual).filter(
                    Actual.home_team == home, Actual.away_team == away, Actual.gameweek == gw
                ).first()
                if act and act.match_id:
                    m['match_id'] = act.match_id
                    continue
                pred = _db_canon.query(Prediction).filter(
                    Prediction.home_team == home, Prediction.away_team == away, Prediction.gameweek == gw
                ).order_by(Prediction.created_at).first()
                if pred and pred.match_id:
                    m['match_id'] = pred.match_id
        finally:
            _db_canon.close()

        # Add form data to each match
        for match in matches:
            match['home_form'] = get_team_form(match['home_team'], up_to_gameweek=gameweek)
            match['away_form'] = get_team_form(match['away_team'], up_to_gameweek=gameweek)
        
        # Separate by status
        upcoming = []
        live = []
        completed = []
        
        for match in matches:
            status = match.get('status', 'SCHEDULED')
            
            if status in ['FINISHED', 'COMPLETE']:
                completed.append(match)
            elif status in ['LIVE', 'IN_PLAY', 'PAUSED']:
                live.append(match)
            else:
                upcoming.append(match)
        
        # For current gameweek, show ALL matches (completed with scores, live, upcoming)
        # For future gameweeks, show only upcoming
        # For past gameweeks, show all matches with final scores
        if gameweek < current_gw:
            # Past gameweek - show all matches with final scores
            return {
                "fixtures": matches,  # Show all matches including completed ones
                "live_matches": [],
                "upcoming_matches": [],
                "completed_matches": completed,
                "gameweek": gameweek,
                "message": f"Gameweek {gameweek} completed. Showing final scores."
            }
        elif gameweek == current_gw:
            # Current gameweek - show ALL matches (completed, live, upcoming)
            all_fixtures = completed + live + upcoming  # Show everything
            return {
                "fixtures": all_fixtures,
                "live_matches": live,
                "upcoming_matches": upcoming,
                "completed_matches": completed,
                "gameweek": gameweek,
                "message": f"{len(completed)} finished, {len(live)} live, {len(upcoming)} upcoming" if (completed or live) else None
            }
        else:
            # Future gameweek - only upcoming
            return {
                "fixtures": upcoming,
                "live_matches": [],
                "upcoming_matches": upcoming,
                "completed_matches": [],
                "gameweek": gameweek
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
async def get_results(gameweek: Optional[int] = None, user: str = Depends(get_current_user)):
    """Get results for a specific gameweek - only shows completed matches"""
    try:
        current_gw = football_api.get_current_gameweek()
        
        if not gameweek:
            gameweek = current_gw
        
        # Only show results for current or past gameweeks
        if gameweek > current_gw:
            return {
                "results": [],
                "gameweek": gameweek,
                "message": f"Gameweek {gameweek} has not started yet. Check Fixtures tab."
            }
        
        # Get all matches from football API for this gameweek
        matches = football_api.get_matches(gameweek=gameweek)
        
        # Fallback to DB (Actual table) if live API returned nothing
        if not matches:
            db_sess = get_db_session()
            try:
                actuals = db_sess.query(Actual).filter(Actual.gameweek == gameweek).all()
                matches = [{
                    'id': a.match_id,
                    'match_id': a.match_id,
                    'gameweek': a.gameweek,
                    'date': a.updated_at.isoformat() if a.updated_at else None,
                    'home_team': a.home_team,
                    'away_team': a.away_team,
                    'home_score': a.actual_home_goals,
                    'away_score': a.actual_away_goals,
                    'status': a.status or ('FINISHED' if a.actual_home_goals is not None else 'SCHEDULED'),
                    'venue': football_api.stadiums.get(a.home_team, 'Stadium')
                } for a in actuals]
            finally:
                db_sess.close()
        
        # Ensure each match has match_id field
        for m in matches:
            if 'match_id' not in m:
                m['match_id'] = str(m.get('id', ''))

        # Resolve canonical match_id: replace FPL/API IDs with the Prediction/Actual
        # table's own match_id so frontend predictions align correctly.
        _db_canon = get_db_session()
        try:
            for m in matches:
                home = m.get('home_team', '')
                away = m.get('away_team', '')
                gw   = m.get('gameweek', gameweek)
                act = _db_canon.query(Actual).filter(
                    Actual.home_team == home, Actual.away_team == away, Actual.gameweek == gw
                ).first()
                if act and act.match_id:
                    m['match_id'] = act.match_id
                    continue
                pred = _db_canon.query(Prediction).filter(
                    Prediction.home_team == home, Prediction.away_team == away, Prediction.gameweek == gw
                ).order_by(Prediction.created_at).first()
                if pred and pred.match_id:
                    m['match_id'] = pred.match_id
        finally:
            _db_canon.close()

        # Add form data to each match
        for match in matches:
            match['home_form'] = get_team_form(match['home_team'], up_to_gameweek=gameweek)
            match['away_form'] = get_team_form(match['away_team'], up_to_gameweek=gameweek)
        
        # For current or past gameweeks, show ALL matches
        # Completed matches will have scores, upcoming will show as scheduled
        completed = [m for m in matches if m.get('status') in ['FINISHED', 'COMPLETE']]
        upcoming = [m for m in matches if m.get('status') not in ['FINISHED', 'COMPLETE', 'LIVE', 'IN_PLAY', 'PAUSED']]
        live = [m for m in matches if m.get('status') in ['LIVE', 'IN_PLAY', 'PAUSED']]
        
        # Show all matches for results view
        all_results = completed + live + upcoming
        
        if not completed and gameweek == current_gw:
            message = "No matches completed yet. " + (f"{len(live)} live, {len(upcoming)} upcoming" if (live or upcoming) else "")
        else:
            message = f"{len(completed)} finished, {len(live)} live, {len(upcoming)} upcoming" if (live or upcoming) else None
        
        return {
            "results": all_results,
            "completed": completed,
            "live": live,
            "upcoming": upcoming,
            "gameweek": gameweek,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_match(request: PredictionRequest, user: str = Depends(get_current_user)):
    """Predict match outcome using enhanced models"""
    try:
        if request.home_team not in CURRENT_SEASON_TEAMS:
            raise HTTPException(status_code=400, detail=f"Invalid home team: {request.home_team}")
        if request.away_team not in CURRENT_SEASON_TEAMS:
            raise HTTPException(status_code=400, detail=f"Invalid away team: {request.away_team}")
        if request.home_team == request.away_team:
            raise HTTPException(status_code=400, detail="Home and away teams must be different")
        
        # Calculate features for both teams
        home_features = calculate_team_features(request.home_team, request.away_team, True, request.gameweek)
        away_features = calculate_team_features(request.away_team, request.home_team, False, request.gameweek)
        
        if home_features is None or away_features is None:
            raise HTTPException(status_code=400, detail="Insufficient match history for prediction")
        
        # Combine features
        all_features = {**home_features, **away_features}
        
        # H2H features - calculate from match history
        h2h_matches = matches_df[
            (((matches_df['home_team'] == request.home_team) & (matches_df['away_team'] == request.away_team)) |
             ((matches_df['home_team'] == request.away_team) & (matches_df['away_team'] == request.home_team)))
        ].tail(10)
        
        if len(h2h_matches) > 0:
            h_wins, a_wins, draws = 0, 0, 0
            h_goals, a_goals = [], []
            for _, m in h2h_matches.iterrows():
                if m['home_team'] == request.home_team:
                    hg, ag = m['home_goals'], m['away_goals']
                else:
                    hg, ag = m['away_goals'], m['home_goals']
                h_goals.append(hg)
                a_goals.append(ag)
                if hg > ag: h_wins += 1
                elif ag > hg: a_wins += 1
                else: draws += 1
            
            all_features['h2h_matches'] = len(h2h_matches)
            all_features['h2h_home_wins'] = h_wins
            all_features['h2h_away_wins'] = a_wins
            all_features['h2h_draws'] = draws
            all_features['h2h_home_goals_avg'] = np.mean(h_goals)
            all_features['h2h_away_goals_avg'] = np.mean(a_goals)
            all_features['h2h_total_goals_avg'] = np.mean(h_goals) + np.mean(a_goals)
            all_features['h2h_home_win_rate'] = h_wins / len(h2h_matches)
        else:
            all_features['h2h_matches'] = 0
            all_features['h2h_home_wins'] = 0
            all_features['h2h_away_wins'] = 0
            all_features['h2h_draws'] = 0
            all_features['h2h_home_goals_avg'] = 1.5
            all_features['h2h_away_goals_avg'] = 1.5
            all_features['h2h_total_goals_avg'] = 3.0
            all_features['h2h_home_win_rate'] = 0.33
        
        # Derived features - V3 (already calculated in home_features)
        all_features['position_diff'] = all_features['home_position'] - all_features['away_position']
        all_features['form_diff'] = all_features['home_form_points_l5'] - all_features['away_form_points_l5']
        all_features['gd_diff'] = all_features['home_gd'] - all_features['away_gd']
        all_features['momentum_home'] = all_features['home_form_points_l5'] - all_features['home_form_points_l10'] / 2
        all_features['momentum_away'] = all_features['away_form_points_l5'] - all_features['away_form_points_l10'] / 2
        
        # Derived features - V2
        all_features['attack_differential_l5'] = all_features['home_form_goals_l5'] - all_features['away_form_conceded_l5']
        all_features['defense_differential_l5'] = all_features['away_form_goals_l5'] - all_features['home_form_conceded_l5']
        all_features['points_differential'] = all_features['home_points'] - all_features['away_points']
        all_features['key_players_differential'] = all_features['home_key_players_out'] - all_features['away_key_players_out']
        all_features['availability_differential'] = all_features['home_availability_rate'] - all_features['away_availability_rate']
        all_features['rest_differential'] = 0

        # Fill in all features that the models were trained on (ELO, attack/defense strength, etc.)
        all_features = fill_missing_score_features(all_features)

        # Prepare feature vector — now contains every feature all three models need
        feature_df = pd.DataFrame([all_features])
        # Safety: add any still-missing columns with 0
        for col in list(base_features) + list(lineup_features) + list(score_features):
            if col not in feature_df.columns:
                feature_df[col] = 0.0

        # STAGE 1: Get predictions from all models independently

        # Base outcome prediction
        base_feature_vector = feature_df[base_features].values
        base_feature_scaled = base_scaler.transform(base_feature_vector)
        base_proba = base_model.predict_proba(base_feature_scaled)[0]

        # Lineup outcome prediction
        lineup_feature_vector = feature_df[lineup_features].values
        lineup_feature_scaled = lineup_scaler.transform(lineup_feature_vector)
        lineup_proba = lineup_model.predict_proba(lineup_feature_scaled)[0]

        # Score prediction (outcome-aware - needs outcome probabilities)
        feature_df['outcome_prob_away']  = float(base_proba[0])
        feature_df['outcome_prob_draw']  = float(base_proba[1])
        feature_df['outcome_prob_home']  = float(base_proba[2])
        feature_df['outcome_prediction'] = int(np.argmax(base_proba))

        feature_df_for_score = feature_df.copy()
        # Ensure every required column exists (fill any remaining gap with 0)

        for col in score_features:
            if col not in feature_df_for_score.columns:
                feature_df_for_score[col] = 0.0

        score_feature_scaled = score_scaler.transform(feature_df_for_score[score_features])
        
        # Predict with ensemble or single model
        if score_is_ensemble:
            # Ensemble prediction with weighted average
            home_preds = []
            away_preds = []
            
            for model_name in ['xgboost', 'lightgbm', 'linear']:
                h_pred = score_model_home[model_name].predict(score_feature_scaled)[0]
                a_pred = score_model_away[model_name].predict(score_feature_scaled)[0]
                home_preds.append(h_pred * score_home_weights[model_name])
                away_preds.append(a_pred * score_away_weights[model_name])
            
            home_goals_pred = sum(home_preds)
            away_goals_pred = sum(away_preds)
        else:
            # Legacy single model prediction
            home_goals_pred = score_model_home.predict(score_feature_scaled)[0]
            away_goals_pred = score_model_away.predict(score_feature_scaled)[0]
        
        # IMPROVED SCORE ROUNDING STRATEGY
        # Use attack/defense strength to adjust predictions
        home_attack_strength = all_features['home_form_goals_l5']
        away_attack_strength = all_features['away_form_goals_l5']
        home_defense_strength = all_features['home_form_conceded_l5']
        away_defense_strength = all_features['away_form_conceded_l5']
        
        # Adjust predictions based on recent form
        home_goals_adjusted = home_goals_pred * (home_attack_strength / 1.5) * (1.5 / away_defense_strength)
        away_goals_adjusted = away_goals_pred * (away_attack_strength / 1.5) * (1.5 / home_defense_strength)
        
        # Smart rounding: use fractional part to determine rounding
        def smart_round(value):
            floor_val = int(np.floor(value))
            frac = value - floor_val
            # If fractional part > 0.4, round up with probability based on fraction
            if frac >= 0.6:
                return floor_val + 1
            elif frac >= 0.3:
                # 50-50 chance based on the fraction
                return floor_val + 1 if np.random.random() < frac else floor_val
            else:
                return floor_val
        
        home_goals = max(0, min(score_max_goals, smart_round(home_goals_adjusted)))
        away_goals = max(0, min(score_max_goals, smart_round(away_goals_adjusted)))
        
        # IMPROVED: Ensure score ALWAYS aligns with ensemble outcome probabilities
        # Calculate ensemble probabilities first to determine final outcome
        avg_home_prob = (base_proba[2] + lineup_proba[2]) / 2
        avg_away_prob = (base_proba[0] + lineup_proba[0]) / 2
        avg_draw_prob = (base_proba[1] + lineup_proba[1]) / 2
        
        # Determine the highest probability outcome
        max_prob = max(avg_home_prob, avg_draw_prob, avg_away_prob)
        
        if avg_home_prob == max_prob and avg_home_prob > 0.40:
            # Ensemble predicts HOME WIN - ensure score reflects this
            if home_goals <= away_goals:
                # Score contradicts - fix it
                if avg_home_prob > 0.60:
                    home_goals = away_goals + 2  # Strong home win
                else:
                    home_goals = away_goals + 1  # Close home win
        
        elif avg_away_prob == max_prob and avg_away_prob > 0.40:
            # Ensemble predicts AWAY WIN - ensure score reflects this
            if away_goals <= home_goals:
                # Score contradicts - fix it
                if avg_away_prob > 0.60:
                    away_goals = home_goals + 2  # Strong away win
                else:
                    away_goals = home_goals + 1  # Close away win
        
        elif avg_draw_prob == max_prob and avg_draw_prob > 0.35:
            # Ensemble predicts DRAW - ensure score reflects this
            if home_goals != away_goals:
                # Score contradicts - make it a draw
                # Use the average of the two scores
                avg_score = int(round((home_goals + away_goals) / 2))
                home_goals = avg_score
                away_goals = avg_score
        
        # STAGE 2: Use Alignment Ensemble for consistent prediction
        # Detect if user provided custom formations (non-default)
        has_custom_formation = (
            request.home_formation is not None and request.home_formation != "4-3-3" or
            request.away_formation is not None and request.away_formation != "4-3-3"
        )
        
        aligned_prediction = ensemble.predict(
            base_proba=base_proba,
            score_pred=(home_goals, away_goals),
            lineup_proba=lineup_proba,
            has_custom_formation=has_custom_formation
        )
        
        # Format with confidence indicators
        formatted_prediction = format_prediction_with_confidence(aligned_prediction)
        
        # Calculate individual model confidence levels and priorities
        base_confidence = float(np.max(base_proba))
        lineup_confidence = float(np.max(lineup_proba))
        score_outcome = 'home_win' if home_goals > away_goals else 'draw' if home_goals == away_goals else 'away_win'
        score_confidence = 0.80 if aligned_prediction['models_aligned'] else 0.60  # Higher if aligned
        
        # Determine if models contradict
        base_pred = 'home_win' if base_proba[2] > max(base_proba[0], base_proba[1]) else 'draw' if base_proba[1] > base_proba[0] else 'away_win'
        lineup_pred = 'home_win' if lineup_proba[2] > max(lineup_proba[0], lineup_proba[1]) else 'draw' if lineup_proba[1] > lineup_proba[0] else 'away_win'
        predictions_list = [base_pred, lineup_pred, score_outcome]
        all_same = len(set(predictions_list)) == 1
        
        # PRIORITY RULES:
        # - With edited lineup: Lineup #1, Base #2, Score #3 (Score never highest)
        # - Default lineup: Base #1, Lineup #2, Score #3 (Score never highest)
        # - When agreeing: Sort by confidence but Score never first
        if not all_same:
            # Models contradict - priority depends on lineup customization
            if has_custom_formation:
                # Edited lineup → Lineup gets highest priority
                model_priorities = {
                    'lineup': {'confidence': lineup_confidence, 'level': 'HIGH' if lineup_confidence >= 0.75 else 'MEDIUM' if lineup_confidence >= 0.60 else 'LOW', 'priority': 1},
                    'base': {'confidence': base_confidence, 'level': 'HIGH' if base_confidence >= 0.75 else 'MEDIUM' if base_confidence >= 0.60 else 'LOW', 'priority': 2},
                    'score': {'confidence': score_confidence, 'level': 'HIGH' if score_confidence >= 0.75 else 'MEDIUM' if score_confidence >= 0.60 else 'LOW', 'priority': 3}
                }
                top_model = 'lineup'
            else:
                # Default lineup → Base gets highest priority
                model_priorities = {
                    'base': {'confidence': base_confidence, 'level': 'HIGH' if base_confidence >= 0.75 else 'MEDIUM' if base_confidence >= 0.60 else 'LOW', 'priority': 1},
                    'lineup': {'confidence': lineup_confidence, 'level': 'HIGH' if lineup_confidence >= 0.75 else 'MEDIUM' if lineup_confidence >= 0.60 else 'LOW', 'priority': 2},
                    'score': {'confidence': score_confidence, 'level': 'HIGH' if score_confidence >= 0.75 else 'MEDIUM' if score_confidence >= 0.60 else 'LOW', 'priority': 3}
                }
                top_model = 'base'
        else:
            # Models agree - sort by confidence but ensure Score is never #1
            model_confidences = [
                ('base', base_confidence),
                ('lineup', lineup_confidence),
                ('score', score_confidence)
            ]
            model_confidences.sort(key=lambda x: x[1], reverse=True)
            
            # If score is ranked first, swap with second place
            if model_confidences[0][0] == 'score':
                model_confidences[0], model_confidences[1] = model_confidences[1], model_confidences[0]
            
            model_priorities = {}
            for idx, (model_name, conf) in enumerate(model_confidences):
                if conf >= 0.75:
                    level = 'HIGH'
                elif conf >= 0.60:
                    level = 'MEDIUM'
                else:
                    level = 'LOW'
                
                model_priorities[model_name] = {
                    'confidence': conf,
                    'level': level,
                    'priority': idx + 1
                }
            top_model = model_confidences[0][0]
        
        # Determine prediction statement based on agreement
        top_confidence = model_priorities[top_model]['confidence']
        
        if all_same and top_confidence >= 0.75:
            prediction_difficulty = 'EASY'
            prediction_statement = f"All models strongly agree - {formatted_prediction['prediction'].replace('_', ' ').title()}"
        elif all_same:
            prediction_difficulty = 'MODERATE'
            prediction_statement = f"All models agree - {formatted_prediction['prediction'].replace('_', ' ').title()}"
        elif aligned_prediction['models_aligned']:
            prediction_difficulty = 'MODERATE'
            prediction_statement = f"Majority consensus - {formatted_prediction['prediction'].replace('_', ' ').title()}"
        else:
            prediction_difficulty = 'DIFFICULT'
            prediction_statement = f"Mixed signals - {formatted_prediction['prediction'].replace('_', ' ').title()} (uncertain)"
        
        # Get standings for context
        standings = calculate_current_standings()
        home_standing = next((s for s in standings if s.team == request.home_team), None)
        away_standing = next((s for s in standings if s.team == request.away_team), None)
        
        # Build comprehensive response with ensemble predictions
        prediction_result = {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "gameweek": request.gameweek or all_features['gameweek'],
            
            # Aligned ensemble prediction (primary)
            "predicted_score": formatted_prediction['expected_score'],
            "outcome_probabilities": formatted_prediction['probabilities'],
            "confidence": formatted_prediction['confidence_display'],
            "agreement": formatted_prediction['agreement_display'],
            
            # BHAKUNDO Priority System
            "prediction_source": aligned_prediction.get('prediction_source', 'Unknown'),
            "bhakundo_verified": aligned_prediction.get('bhakundo_verified', False),
            "model_used": aligned_prediction.get('model_used', 'weighted'),
            
            # Model priorities and individual confidence
            "model_priorities": model_priorities,
            "prediction_difficulty": prediction_difficulty,
            "prediction_statement": prediction_statement,
            "top_priority_model": top_model,
            
            # Individual model outputs (for transparency)
            "base_outcome_probabilities": {
                "away_win": float(base_proba[0]),
                "draw": float(base_proba[1]),
                "home_win": float(base_proba[2])
            },
            "lineup_outcome_probabilities": {
                "away_win": float(lineup_proba[0]),
                "draw": float(lineup_proba[1]),
                "home_win": float(lineup_proba[2])
            },
            "score_outcome": score_outcome,
            
            # Context
            "home_standing": {
                "position": home_standing.position if home_standing else None,
                "points": home_standing.points if home_standing else None,
                "form": home_standing.form if home_standing else []
            },
            "away_standing": {
                "position": away_standing.position if away_standing else None,
                "points": away_standing.points if away_standing else None,
                "form": away_standing.form if away_standing else []
            },
            "home_stats": {
                "ppg": round(all_features['home_ppg_l10'], 2),
                "avg_goals_scored": round(all_features['home_avg_goals_scored_l10'], 2),
                "avg_goals_conceded": round(all_features['home_avg_goals_conceded_l10'], 2),
                "win_rate": round(all_features['home_win_rate_l10'] * 100, 1),
                "availability": round(all_features['home_availability_rate'] * 100, 1)
            },
            "away_stats": {
                "ppg": round(all_features['away_ppg_l10'], 2),
                "avg_goals_scored": round(all_features['away_avg_goals_scored_l10'], 2),
                "avg_goals_conceded": round(all_features['away_avg_goals_conceded_l10'], 2),
                "win_rate": round(all_features['away_win_rate_l10'] * 100, 1),
                "availability": round(all_features['away_availability_rate'] * 100, 1)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Store prediction in database
        try:
            db = get_db_session()
            gw = request.gameweek or all_features['gameweek']
            verdict_text = prediction_statement

            canonical_mid = str(request.match_id) if request.match_id else None

            # ── Step 1: ensure an Actual row exists (FK requires it) ──────────
            if canonical_mid:
                # Try to find existing Actual by match_id first
                actual_row = db.query(Actual).filter(Actual.match_id == canonical_mid).first()
                if not actual_row:
                    # Also check by teams+gameweek (in case a different match_id exists)
                    actual_row = db.query(Actual).filter(
                        Actual.home_team == request.home_team,
                        Actual.away_team == request.away_team,
                        Actual.gameweek == gw,
                    ).first()
                if actual_row:
                    # Use the existing Actual row's match_id as canonical
                    canonical_mid = actual_row.match_id
                else:
                    # Create a placeholder SCHEDULED Actual row so FK is satisfied
                    actual_row = Actual(
                        match_id=canonical_mid,
                        home_team=request.home_team,
                        away_team=request.away_team,
                        gameweek=gw,
                        season="2025-26",
                        status="SCHEDULED",
                        updated_at=datetime.utcnow(),
                    )
                    db.add(actual_row)
                    db.flush()   # write to DB transaction so FK is visible

            # ── Step 2: upsert Prediction row ────────────────────────────────
            prediction_record = (
                db.query(Prediction).filter(Prediction.match_id == canonical_mid).first()
                if canonical_mid else None
            )
            if not prediction_record:
                # Also try matching by teams + gameweek (handles id mismatches)
                prediction_record = db.query(Prediction).filter(
                    Prediction.home_team == request.home_team,
                    Prediction.away_team == request.away_team,
                    Prediction.gameweek == gw,
                ).first()

            prediction_fields = dict(
                match_id=canonical_mid,
                home_team=request.home_team,
                away_team=request.away_team,
                gameweek=gw,
                season="2025-26",
                predicted_outcome=formatted_prediction['prediction'].replace('_', ' ').title(),
                predicted_home_goals=home_goals,
                predicted_away_goals=away_goals,
                base_home_prob=float(base_proba[2]),
                base_draw_prob=float(base_proba[1]),
                base_away_prob=float(base_proba[0]),
                lineup_home_prob=float(lineup_proba[2]),
                lineup_draw_prob=float(lineup_proba[1]),
                lineup_away_prob=float(lineup_proba[0]),
                ensemble_home_prob=float(avg_home_prob),
                ensemble_draw_prob=float(avg_draw_prob),
                ensemble_away_prob=float(avg_away_prob),
                ensemble_method=aligned_prediction.get('prediction_source', 'Unknown'),
                confidence=float(top_confidence),
                verdict=verdict_text,
                updated_at=datetime.utcnow(),
            )

            if prediction_record:
                # UPDATE existing row
                for k, v in prediction_fields.items():
                    setattr(prediction_record, k, v)
            else:
                # INSERT new row
                prediction_record = Prediction(**prediction_fields)
                db.add(prediction_record)

            db.commit()
            prediction_result["prediction_id"] = prediction_record.id
            db.close()
        except Exception as db_error:
            print(f"Warning: Could not save to database: {db_error}")
            try:
                db.rollback()
                db.close()
            except Exception:
                pass
            # Continue without failing — prediction result is still returned
        
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/save-prediction/{match_id}")
async def save_prediction(match_id: str, prediction: dict, user: str = Depends(get_current_user)):
    """Save a prediction for a match before it starts"""
    try:
        # Add timestamp if not present
        if 'timestamp' not in prediction:
            prediction['timestamp'] = datetime.now().isoformat()
        
        save_prediction_to_file(match_id, prediction)
        return {"status": "success", "message": "Prediction saved", "match_id": match_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving prediction: {str(e)}")


@app.get("/get-prediction/{match_id}")
async def get_prediction(match_id: str, user: str = Depends(get_current_user)):
    """Get saved prediction for a match"""
    try:
        prediction = get_saved_prediction(match_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="No prediction found for this match")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prediction: {str(e)}")


@app.get("/predictions/all")
async def get_all_predictions(user: str = Depends(get_current_user)):
    """Get all saved predictions"""
    try:
        predictions = load_saved_predictions()
        return {"predictions": predictions, "total": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving predictions: {str(e)}")


@app.get("/prediction-history")
async def get_prediction_history(
    limit: int = 50,
    team: Optional[str] = None,
    gameweek: Optional[int] = None,
    user: str = Depends(get_current_user)
):
    """Get prediction history from database"""
    try:
        db = get_db_session()
        query = db.query(Prediction)
        
        # Filter by team if specified
        if team:
            query = query.filter(
                (Prediction.home_team == team) |
                (Prediction.away_team == team)
            )
        
        # Filter by gameweek if specified
        if gameweek:
            query = query.filter(Prediction.gameweek == gameweek)
        
        # Order by most recent first
        predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
        
        # Fetch corresponding Actual records in bulk
        match_ids = [p.match_id for p in predictions if p.match_id]
        actuals_map = {}
        if match_ids:
            for act in db.query(Actual).filter(Actual.match_id.in_(match_ids)).all():
                actuals_map[act.match_id] = act
        
        result = []
        for pred in predictions:
            act = actuals_map.get(pred.match_id) if pred.match_id else None
            result.append({
                "id": pred.id,
                "match_id": pred.match_id,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "gameweek": pred.gameweek,
                "season": pred.season,
                "predicted_outcome": pred.predicted_outcome,
                "predicted_score": f"{pred.predicted_home_goals}-{pred.predicted_away_goals}",
                "predicted_home_goals": pred.predicted_home_goals,
                "predicted_away_goals": pred.predicted_away_goals,
                "confidence": pred.confidence,
                "ensemble_method": pred.ensemble_method,
                "created_at": pred.created_at.isoformat(),
                "actual_result": {
                    "score": f"{act.actual_home_goals}-{act.actual_away_goals}" if act and act.actual_home_goals is not None else None,
                    "outcome": act.actual_outcome if act else None
                } if act else None
            })
        
        db.close()
        return {
            "predictions": result,
            "total": len(result),
            "filters": {
                "team": team,
                "gameweek": gameweek,
                "limit": limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prediction history: {str(e)}")


@app.get("/saved-prediction/{match_id}")
async def get_saved_prediction_by_match(match_id: str, user: str = Depends(get_current_user)):
    """Get the earliest saved prediction for a specific match (first prediction made before match)"""
    try:
        db = get_db_session()
        
        # Get the prediction for this match_id
        prediction = db.query(Prediction).filter(
            Prediction.match_id == match_id
        ).order_by(Prediction.created_at.asc()).first()
        
        if not prediction:
            db.close()
            raise HTTPException(status_code=404, detail=f"No saved prediction found for match {match_id}")
        
        # Get actual result if it exists
        actual = db.query(Actual).filter(Actual.match_id == match_id).first()
        db.close()
        
        # Use saved ensemble probabilities
        ensemble_home = prediction.ensemble_home_prob or prediction.base_home_prob
        ensemble_draw = prediction.ensemble_draw_prob or prediction.base_draw_prob
        ensemble_away = prediction.ensemble_away_prob or prediction.base_away_prob
        
        return {
            "prediction": {
                "id": prediction.id,
                "match_id": prediction.match_id,
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "gameweek": prediction.gameweek,
                "predicted_outcome": prediction.predicted_outcome,
                "predicted_score": {
                    "home": prediction.predicted_home_goals,
                    "away": prediction.predicted_away_goals
                },
                "outcome_probabilities": {
                    "home_win": ensemble_home,
                    "draw": ensemble_draw,
                    "away_win": ensemble_away
                },
                "base_outcome_probabilities": {
                    "home_win": prediction.base_home_prob,
                    "draw": prediction.base_draw_prob,
                    "away_win": prediction.base_away_prob
                },
                "lineup_outcome_probabilities": {
                    "home_win": prediction.lineup_home_prob,
                    "draw": prediction.lineup_draw_prob,
                    "away_win": prediction.lineup_away_prob
                },
                "confidence": prediction.confidence,
                "ensemble_method": prediction.ensemble_method,
                "created_at": prediction.created_at.isoformat(),
                "actual_result": {
                    "home_goals": actual.actual_home_goals,
                    "away_goals": actual.actual_away_goals,
                    "outcome": actual.actual_outcome
                } if actual else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving saved prediction: {str(e)}")


@app.get("/gameweek-predictions/{gameweek}")
async def get_gameweek_predictions(gameweek: int, user: str = Depends(get_current_user)):
    """Get all earliest saved predictions for a specific gameweek"""
    try:
        db = get_db_session()
        
        # Get one prediction per match_id for this gameweek
        predictions = db.query(Prediction).filter(
            Prediction.gameweek == gameweek,
            Prediction.match_id.isnot(None)
        ).all()
        
        # Deduplicate: keep one per match_id (the one with earliest created_at)
        seen = {}
        for pred in predictions:
            if pred.match_id not in seen or pred.created_at < seen[pred.match_id].created_at:
                seen[pred.match_id] = pred
        predictions = list(seen.values())
        
        # Fetch corresponding Actual records
        match_ids = [p.match_id for p in predictions]
        actuals_map = {}
        if match_ids:
            for act in db.query(Actual).filter(Actual.match_id.in_(match_ids)).all():
                actuals_map[act.match_id] = act
        
        db.close()
        
        result = {}
        for pred in predictions:
            if pred.match_id:
                act = actuals_map.get(pred.match_id)
                ensemble_home = pred.ensemble_home_prob or pred.base_home_prob
                ensemble_draw = pred.ensemble_draw_prob or pred.base_draw_prob
                ensemble_away = pred.ensemble_away_prob or pred.base_away_prob
                
                result[pred.match_id] = {
                    "id": pred.id,
                    "match_id": pred.match_id,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "gameweek": pred.gameweek,
                    "predicted_outcome": pred.predicted_outcome,
                    "predicted_score": {
                        "home": pred.predicted_home_goals,
                        "away": pred.predicted_away_goals
                    },
                    "outcome_probabilities": {
                        "home_win": ensemble_home,
                        "draw": ensemble_draw,
                        "away_win": ensemble_away
                    },
                    "base_outcome_probabilities": {
                        "home_win": pred.base_home_prob,
                        "draw": pred.base_draw_prob,
                        "away_win": pred.base_away_prob
                    },
                    "lineup_outcome_probabilities": {
                        "home_win": pred.lineup_home_prob,
                        "draw": pred.lineup_draw_prob,
                        "away_win": pred.lineup_away_prob
                    },
                    "confidence": pred.confidence,
                    "ensemble_method": pred.ensemble_method,
                    "created_at": pred.created_at.isoformat(),
                    "actual_result": {
                        "home_goals": act.actual_home_goals if act else None,
                        "away_goals": act.actual_away_goals if act else None,
                        "outcome": act.actual_outcome if act else None
                    } if act else None
                }
        
        return {
            "gameweek": gameweek,
            "predictions": result,
            "total": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving gameweek predictions: {str(e)}")


@app.get("/prediction-stats")
async def get_prediction_stats(user: str = Depends(get_current_user)):
    """Get prediction accuracy statistics"""
    try:
        db = get_db_session()
        
        # Get all Actual records with outcome, joined with Prediction for correctness check
        completed_actuals = db.query(Actual).filter(
            Actual.actual_outcome.isnot(None)
        ).all()
        
        if not completed_actuals:
            return {
                "total_predictions": 0,
                "outcome_accuracy": 0,
                "message": "No completed matches to analyze yet"
            }
        
        # Fetch corresponding predictions for correctness + ensemble_method grouping
        act_match_ids = [a.match_id for a in completed_actuals if a.match_id]
        preds_map = {}
        if act_match_ids:
            for p in db.query(Prediction).filter(Prediction.match_id.in_(act_match_ids)).all():
                preds_map[p.match_id] = p
        
        total = len(completed_actuals)
        correct_outcomes = sum(
            1 for a in completed_actuals
            if preds_map.get(a.match_id) and preds_map[a.match_id].predicted_outcome == a.actual_outcome
        )
        outcome_accuracy = (correct_outcomes / total * 100) if total > 0 else 0
        
        # Group by ensemble method
        by_method = {}
        for act in completed_actuals:
            pred = preds_map.get(act.match_id)
            method = (pred.ensemble_method if pred else None) or "unknown"
            if method not in by_method:
                by_method[method] = {"total": 0, "correct_outcomes": 0}
            by_method[method]["total"] += 1
            if pred and pred.predicted_outcome == act.actual_outcome:
                by_method[method]["correct_outcomes"] += 1
        
        # Calculate accuracy by method
        method_accuracy = {}
        for method, stats in by_method.items():
            method_accuracy[method] = {
                "total": stats["total"],
                "correct_outcomes": stats["correct_outcomes"],
                "outcome_accuracy": round(stats["correct_outcomes"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0
            }
        
        db.close()
        return {
            "total_predictions": total,
            "correct_outcomes": correct_outcomes,
            "outcome_accuracy": round(outcome_accuracy, 2),
            "by_method": method_accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")


class ContactMessage(BaseModel):
    """Contact form message"""
    email: EmailStr
    message: str


@app.post("/send-contact")
async def send_contact_message(contact: ContactMessage, user: str = Depends(get_current_user)):
    """Send contact form message to admin email"""
    try:
        # Email configuration
        recipient_email = "bs426808@gmail.com"
        sender_email = "noreply@bhakundo.com"  # This will be overridden by Gmail
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Bhakundo Contact Form - Message from {contact.email}"
        msg['From'] = contact.email
        msg['To'] = recipient_email
        
        # Create email body
        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
              <h2 style="color: #3B82F6; border-bottom: 2px solid #3B82F6; padding-bottom: 10px;">New Contact Form Message</h2>
              
              <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p style="margin: 5px 0;"><strong>From:</strong> {contact.email}</p>
                <p style="margin: 5px 0;"><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
              </div>
              
              <div style="margin: 20px 0;">
                <h3 style="color: #3B82F6;">Message:</h3>
                <div style="background-color: #ffffff; padding: 15px; border-left: 4px solid #3B82F6; border-radius: 5px;">
                  <p style="white-space: pre-wrap;">{contact.message}</p>
                </div>
              </div>
              
              <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
              <p style="color: #666; font-size: 12px; text-align: center;">
                This message was sent from the Bhakundo Premier League Predictor contact form.
              </p>
            </div>
          </body>
        </html>
        """
        
        text_content = f"""
        New Contact Form Message from Bhakundo Premier League Predictor
        
        From: {contact.email}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Message:
        {contact.message}
        
        ---
        This message was sent from the Bhakundo Premier League Predictor contact form.
        """
        
        # Attach both plain text and HTML versions
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Try to send email using Gmail's SMTP (requires app password or OAuth2)
        # Note: For production, you'll need to configure Gmail app password or use a transactional email service
        try:
            # Using local SMTP server or Gmail (you'll need to configure credentials)
            # For now, we'll just log and return success
            print(f"\n{'='*60}")
            print(f"📧 NEW CONTACT FORM MESSAGE")
            print(f"{'='*60}")
            print(f"From: {contact.email}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nMessage:\n{contact.message}")
            print(f"{'='*60}\n")
            
            # For production: Uncomment and configure with Gmail app password
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login('your_gmail@gmail.com', 'your_app_password')
            # server.send_message(msg)
            # server.quit()
            
            return {
                "success": True,
                "message": "Message sent successfully! We'll get back to you soon.",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as email_error:
            print(f"⚠️  Email sending failed: {str(email_error)}")
            print(f"Message logged locally from {contact.email}")
            # Still return success since message is logged
            return {
                "success": True,
                "message": "Message received and logged. We'll get back to you soon.",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"❌ Error processing contact form: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")


@app.post("/update-prediction-results")
async def update_prediction_results(gameweek: Optional[int] = None, user: str = Depends(get_current_user)):
    """
    Fetch finished matches from the Football API and upsert scores into the actual table.
    Matches by (home_team, away_team, gameweek) so match_id source doesn't matter.
    """
    try:
        db = get_db_session()
        current_gw = football_api.get_current_gameweek()
        gameweeks_to_sync = [gameweek] if gameweek else list(range(max(1, current_gw - 4), current_gw + 1))
        result = football_api.sync_actual_from_api(db, gameweeks=gameweeks_to_sync)
        db.close()
        return {
            "status": "success",
            "message": f"Synced actual results for gameweeks {gameweeks_to_sync}",
            "updated": result["updated"],
            "inserted": result["inserted"],
            "errors": result["errors"],
            "gameweeks_synced": gameweeks_to_sync,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating results: {str(e)}")


@app.get("/sync-all-results")
async def sync_all_results(user: str = Depends(get_current_user)):
    """
    Sync actual results for all tracked gameweeks (GW1 → current) from the Football API.
    """
    try:
        db = get_db_session()
        current_gw = football_api.get_current_gameweek()
        all_gameweeks = list(range(1, current_gw + 1))
        result = football_api.sync_actual_from_api(db, gameweeks=all_gameweeks)
        db.close()
        return {
            "status": "success",
            "message": f"Synced GW1-{current_gw}",
            "updated": result["updated"],
            "inserted": result["inserted"],
            "errors": result["errors"],
            "gameweeks_synced": len(all_gameweeks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing results: {str(e)}")


@app.get("/game-of-gw")
async def get_game_of_the_gw(user: str = Depends(get_current_user)):
    """
    Return the latest active Game of the GW entry, joined with prediction & actual tables.
    Table: game_of_the_gw
    """
    from sqlalchemy import text as sa_text
    try:
        db = get_db_session()
        row = db.execute(sa_text("""
            SELECT
                m.id            AS gotgw_id,
                m.match_id,
                m.gameweek,
                m.match_date,
                m.home_team,
                m.away_team,
                m.reason,
                p.predicted_home_goals,
                p.predicted_away_goals,
                p.predicted_outcome,
                p.ensemble_home_prob,
                p.ensemble_draw_prob,
                p.ensemble_away_prob,
                p.confidence,
                p.verdict,
                a.actual_home_goals,
                a.actual_away_goals,
                a.status
            FROM game_of_the_gw m
            LEFT JOIN prediction p ON p.match_id = m.match_id
            LEFT JOIN actual     a ON a.match_id = m.match_id
            WHERE m.is_active = true
            ORDER BY m.id DESC
            LIMIT 1
        """)).fetchone()
        db.close()

        if not row:
            raise HTTPException(status_code=404, detail="No active game of the GW found")

        data = dict(row._mapping)
        if data.get("match_date") is not None:
            data["match_date"] = data["match_date"].isoformat()
        return data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Game of the GW: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Premier League Predictor API 2025-26...")
    print(f"📊 Models: Enhanced v3 (125 combined features)")
    print(f"⚽ Teams: {len(CURRENT_SEASON_TEAMS)}")
    print("📝 Manual retraining mode: Retrain locally and redeploy")
    uvicorn.run(app, host="0.0.0.0", port=8000)
