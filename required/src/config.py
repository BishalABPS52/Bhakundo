"""Premier League Prediction System - Configuration Module"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    football_data_api_key: str = os.getenv("FOOTBALL_DATA_API_KEY", "")
    rapid_api_key: str = os.getenv("RAPID_API_KEY", "")
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    
    # Database
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "pl_predictions")
    db_user: str = os.getenv("DB_USER", "pl_user")
    db_password: str = os.getenv("DB_PASSWORD", "pl_secure_password_2025")
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    # Model Settings
    model_path: Path = Path(os.getenv("MODEL_PATH", "data/models/"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    
    # API Settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Premier League Team Information (2025-26 Season)
PREMIER_LEAGUE_TEAMS = {
    "Arsenal": {"id": 2, "short_name": "ARS", "stadium": "Emirates Stadium"},
    "Manchester City": {"id": 6, "short_name": "MCI", "stadium": "Etihad Stadium"},
    "Aston Villa": {"id": 3, "short_name": "AVL", "stadium": "Villa Park"},
    "Chelsea": {"id": 4, "short_name": "CHE", "stadium": "Stamford Bridge"},
    "Manchester United": {"id": 5, "short_name": "MUN", "stadium": "Old Trafford"},
    "Liverpool": {"id": 1, "short_name": "LIV", "stadium": "Anfield"},
    "Sunderland": {"id": 7, "short_name": "SUN", "stadium": "Stadium of Light"},
    "Crystal Palace": {"id": 8, "short_name": "CRY", "stadium": "Selhurst Park"},
    "Brighton": {"id": 9, "short_name": "BHA", "stadium": "American Express Stadium"},
    "Everton": {"id": 10, "short_name": "EVE", "stadium": "Goodison Park"},
    "Newcastle": {"id": 11, "short_name": "NEW", "stadium": "St James' Park"},
    "Brentford": {"id": 12, "short_name": "BRE", "stadium": "Gtech Community Stadium"},
    "Fulham": {"id": 13, "short_name": "FUL", "stadium": "Craven Cottage"},
    "Tottenham": {"id": 14, "short_name": "TOT", "stadium": "Tottenham Hotspur Stadium"},
    "Bournemouth": {"id": 15, "short_name": "BOU", "stadium": "Vitality Stadium"},
    "Leeds United": {"id": 16, "short_name": "LEE", "stadium": "Elland Road"},
    "Nottingham Forest": {"id": 17, "short_name": "NFO", "stadium": "City Ground"},
    "West Ham": {"id": 18, "short_name": "WHU", "stadium": "London Stadium"},
    "Burnley": {"id": 19, "short_name": "BUR", "stadium": "Turf Moor"},
    "Wolves": {"id": 20, "short_name": "WOL", "stadium": "Molineux Stadium"},
}

# API Endpoints
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
UNDERSTAT_BASE_URL = "https://understat.com"

# Model Configuration
MODEL_CONFIG = {
    "outcome_model": {
        "xgboost": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 8,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
        },
        "lightgbm": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "num_leaves": 63,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
        },
        "catboost": {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 8,
            "loss_function": "MultiClass",
            "random_state": 42,
            "verbose": False,
        },
    },
    "ensemble_weights": {
        "xgboost": 0.35,
        "lightgbm": 0.35,
        "catboost": 0.30,
    },
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "form_windows": [5, 10, 15],  # Last N games for form calculation
    "xg_windows": [5, 10],  # Last N games for xG metrics
    "max_h2h_matches": 10,  # Maximum H2H matches to consider
    "min_matches_for_features": 5,  # Minimum matches required for feature calculation
}

# Data Collection Configuration
DATA_COLLECTION_CONFIG = {
    "seasons": ["2020", "2021", "2022", "2023", "2024", "2025"],  # Updated to include 2025
    "rate_limits": {
        "football_data": 10,  # requests per minute
        "fbref": 3,  # seconds between requests
        "understat": 2,  # seconds between requests
    },
}

# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
