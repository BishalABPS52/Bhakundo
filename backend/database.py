"""
Database module for storing prediction history
Uses SQLAlchemy with PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from backend/.env (works regardless of cwd)
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pl_user:pl_secure_password_2025@localhost:5432/pl_predictions"
)

# Fix for Render's postgres:// URL (needs to be postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class Prediction(Base):
    """Store match predictions - one prediction per match"""
    __tablename__ = "prediction"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(100), unique=True, nullable=False, index=True)  # Unique match identifier
    
    # Match details
    home_team = Column(String(100), nullable=False, index=True)
    away_team = Column(String(100), nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)
    match_date = Column(DateTime, nullable=True)
    season = Column(String(20), default="2025-26", nullable=False)
    
    # Predicted scores
    predicted_home_goals = Column(Integer, nullable=False)
    predicted_away_goals = Column(Integer, nullable=False)
    predicted_outcome = Column(String(20), nullable=False)  # Home Win, Draw, Away Win
    
    # Base model probabilities
    base_home_prob = Column(Float, nullable=False)
    base_draw_prob = Column(Float, nullable=False)
    base_away_prob = Column(Float, nullable=False)
    
    # Lineup model probabilities
    lineup_home_prob = Column(Float, nullable=False)
    lineup_draw_prob = Column(Float, nullable=False)
    lineup_away_prob = Column(Float, nullable=False)
    
    # Final ensemble probabilities
    ensemble_home_prob = Column(Float, nullable=False)
    ensemble_draw_prob = Column(Float, nullable=False)
    ensemble_away_prob = Column(Float, nullable=False)
    
    # Verdict (confidence and method)
    confidence = Column(Float, nullable=False)  # 0-1 confidence score
    ensemble_method = Column(String(50))  # Agreement method used
    verdict = Column(String(200))  # Human-readable verdict: "All models strongly agree - Home Win"
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Prediction {self.match_id}: {self.home_team} vs {self.away_team} - {self.predicted_outcome}>"


class Actual(Base):
    """Store actual match results"""
    __tablename__ = "actual"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(100), unique=True, nullable=False, index=True)  # Links to Prediction table
    
    # Match details
    home_team = Column(String(100), nullable=False, index=True)
    away_team = Column(String(100), nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)
    match_date = Column(DateTime, nullable=True)
    season = Column(String(20), default="2025-26", nullable=False)
    
    # Actual scores (NULL until match is completed)
    actual_home_goals = Column(Integer, nullable=True)
    actual_away_goals = Column(Integer, nullable=True)
    actual_outcome = Column(String(20), nullable=True)  # Home Win, Draw, Away Win
    
    # Match status
    status = Column(String(20), default="SCHEDULED")  # SCHEDULED, IN_PLAY, FINISHED, POSTPONED
    
    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, index=True)
    last_api_update = Column(DateTime)  # When we last fetched from Football API
    
    def __repr__(self):
        return f"<Actual {self.match_id}: {self.home_team} {self.actual_home_goals}-{self.actual_away_goals} {self.away_team}>"


class Standing(Base):
    """Store Premier League standings - updated periodically"""
    __tablename__ = "standing"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Season and Gameweek
    season = Column(String(20), default="2025-26", nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)  # Last completed gameweek
    
    # Team information
    team = Column(String(100), nullable=False, index=True)
    position = Column(Integer, nullable=False)
    
    # Statistics
    played = Column(Integer, nullable=False)
    won = Column(Integer, nullable=False)
    drawn = Column(Integer, nullable=False)
    lost = Column(Integer, nullable=False)
    goals_for = Column(Integer, nullable=False)
    goals_against = Column(Integer, nullable=False)
    goal_difference = Column(Integer, nullable=False)
    points = Column(Integer, nullable=False)
    
    # Form (last 5 matches)
    form = Column(String(20), nullable=True)  # e.g., "W,W,D,L,W"
    
    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, index=True)
    source = Column(String(50), default="api")  # 'api' or 'calculated'
    
    def __repr__(self):
        return f"<Standing {self.position}. {self.team} - {self.points}pts (GW{self.gameweek})>"


def init_db():
    """Initialize database and create tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session (for non-FastAPI usage)"""
    return SessionLocal()
