"""Database models and schema definitions using SQLAlchemy ORM"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Time,
    ForeignKey, Boolean, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import settings

Base = declarative_base()


class Team(Base):
    """Teams table"""
    __tablename__ = "teams"
    
    team_id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String(100), nullable=False, unique=True)
    short_name = Column(String(10), nullable=False)
    api_football_id = Column(Integer)
    fbref_id = Column(String(20))
    understat_id = Column(String(50))
    stadium_name = Column(String(100))
    founded_year = Column(Integer)
    
    # Relationships
    home_matches = relationship("Match", back_populates="home_team", foreign_keys="Match.home_team_id")
    away_matches = relationship("Match", back_populates="away_team", foreign_keys="Match.away_team_id")
    players = relationship("Player", back_populates="team")
    form_records = relationship("TeamForm", back_populates="team")


class Match(Base):
    """Matches table - historical match data"""
    __tablename__ = "matches"
    
    match_id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String(10), nullable=False)  # e.g., "2023-24"
    gameweek = Column(Integer, nullable=False)
    match_date = Column(DateTime, nullable=False)
    
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    
    # Match Result
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(String(1))  # 'H', 'D', 'A'
    half_time_home_goals = Column(Integer)
    half_time_away_goals = Column(Integer)
    
    # Expected Goals
    home_xg = Column(Float)
    away_xg = Column(Float)
    
    # Match Statistics
    home_possession = Column(Integer)  # Percentage
    away_possession = Column(Integer)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)
    
    # Additional Information
    referee_name = Column(String(100))
    attendance = Column(Integer)
    weather_temp = Column(Float)
    weather_condition = Column(String(50))
    kickoff_time = Column(Time)
    
    # Relationships
    home_team = relationship("Team", back_populates="home_matches", foreign_keys=[home_team_id])
    away_team = relationship("Team", back_populates="away_matches", foreign_keys=[away_team_id])
    player_stats = relationship("PlayerMatchStats", back_populates="match")


class Player(Base):
    """Players table"""
    __tablename__ = "players"
    
    player_id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"))
    position = Column(String(20))  # GK, DEF, MID, FWD
    nationality = Column(String(50))
    age = Column(Integer)
    market_value = Column(Float)
    
    # External IDs
    api_football_id = Column(Integer)
    fbref_id = Column(String(20))
    understat_id = Column(String(20))
    fpl_id = Column(Integer)
    
    # Relationships
    team = relationship("Team", back_populates="players")
    match_stats = relationship("PlayerMatchStats", back_populates="player")
    injuries = relationship("Injury", back_populates="player")


class PlayerMatchStats(Base):
    """Player performance statistics for each match"""
    __tablename__ = "player_match_stats"
    
    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    
    # Playing Time
    minutes_played = Column(Integer)
    started = Column(Boolean, default=False)
    
    # Performance Metrics
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    key_passes = Column(Integer, default=0)
    dribbles_attempted = Column(Integer, default=0)
    dribbles_completed = Column(Integer, default=0)
    
    # Defensive Metrics
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    clearances = Column(Integer, default=0)
    
    # Disciplinary
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    
    # Advanced Metrics
    rating = Column(Float)  # Match rating (e.g., 7.5/10)
    xg = Column(Float)  # Player xG
    xa = Column(Float)  # Player xA (expected assists)
    
    # Relationships
    match = relationship("Match", back_populates="player_stats")
    player = relationship("Player", back_populates="match_stats")


class TeamForm(Base):
    """Team form metrics over time"""
    __tablename__ = "team_form"
    
    form_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    date = Column(Date, nullable=False)
    
    # Last 5 Games
    last_5_form = Column(String(10))  # e.g., 'WWDLW'
    last_5_points = Column(Integer)
    last_5_goals_scored = Column(Integer)
    last_5_goals_conceded = Column(Integer)
    
    # Last 10 Games
    last_10_form = Column(String(20))
    last_10_points = Column(Integer)
    ppg_last_10 = Column(Float)  # Points per game
    clean_sheets_last_10 = Column(Integer)
    
    # Home/Away Split
    home_form_last_5 = Column(String(10))
    away_form_last_5 = Column(String(10))
    
    # Relationships
    team = relationship("Team", back_populates="form_records")


class Injury(Base):
    """Player injuries and suspensions"""
    __tablename__ = "injuries"
    
    injury_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    
    injury_type = Column(String(100))  # e.g., "Hamstring Strain"
    injury_date = Column(Date, nullable=False)
    expected_return = Column(Date)
    status = Column(String(20))  # 'Injured', 'Doubtful', 'Suspended', 'Recovered'
    source = Column(String(100))  # Data source
    
    # Relationships
    player = relationship("Player", back_populates="injuries")


class Fixture(Base):
    """Upcoming fixtures"""
    __tablename__ = "fixtures"
    
    fixture_id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String(10), nullable=False)
    gameweek = Column(Integer, nullable=False)
    match_date = Column(DateTime, nullable=False)
    
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    
    # Betting Odds
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    
    # Fixture Difficulty
    fdr_home = Column(Integer)  # FPL Fixture Difficulty Rating (1-5)
    fdr_away = Column(Integer)


class H2HHistory(Base):
    """Head-to-head history between teams"""
    __tablename__ = "h2h_history"
    
    h2h_id = Column(Integer, primary_key=True, autoincrement=True)
    team1_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    team2_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    
    # Last 10 Meetings
    last_10_team1_wins = Column(Integer)
    last_10_draws = Column(Integer)
    last_10_team2_wins = Column(Integer)
    
    # Goal Averages
    avg_goals_team1 = Column(Float)
    avg_goals_team2 = Column(Float)
    
    # Last Updated
    last_updated = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Stored predictions for tracking and evaluation"""
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.fixture_id"), nullable=False)
    
    # Prediction Timestamp
    predicted_at = Column(DateTime, default=datetime.utcnow)
    
    # Outcome Probabilities
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    predicted_outcome = Column(String(1))  # 'H', 'D', 'A'
    
    # Score Prediction
    predicted_home_goals = Column(Integer)
    predicted_away_goals = Column(Integer)
    predicted_scoreline = Column(String(10))  # e.g., "2-1"
    
    # Confidence & Metadata
    confidence_level = Column(String(20))  # 'High', 'Medium', 'Low'
    model_version = Column(String(20))
    
    # Actual Result (filled after match)
    actual_home_goals = Column(Integer)
    actual_away_goals = Column(Integer)
    actual_result = Column(String(1))
    prediction_correct = Column(Boolean)


# Database connection and session management
def get_engine():
    """Create database engine"""
    return create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        echo=False
    )


def get_session():
    """Create database session"""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db():
    """Initialize database - create all tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully!")


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("All tables dropped!")


if __name__ == "__main__":
    # Create all tables
    init_db()
