"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class MatchInfo(BaseModel):
    """Match information"""
    home_team: str
    away_team: str
    match_date: Optional[str] = None


class OutcomePrediction(BaseModel):
    """Match outcome prediction"""
    home_win_probability: float = Field(..., ge=0, le=1)
    draw_probability: float = Field(..., ge=0, le=1)
    away_win_probability: float = Field(..., ge=0, le=1)
    predicted_outcome: str
    confidence: str


class ScorelinePrediction(BaseModel):
    """Individual scoreline prediction"""
    scoreline: str
    probability: float
    probability_pct: float


class ScorePrediction(BaseModel):
    """Score prediction details"""
    most_likely_score: str
    home_goals: int
    away_goals: int
    probability: float
    top_5_scorelines: List[Dict]


class AdditionalPredictions(BaseModel):
    """Additional betting market predictions"""
    over_2_5: float
    under_2_5: float
    btts_yes: float
    btts_no: float


class KeyFactors(BaseModel):
    """Key factors influencing prediction"""
    home_recent_form: str
    away_recent_form: str
    home_xg_avg: float
    away_xg_avg: float


class MatchPrediction(BaseModel):
    """Complete match prediction"""
    match_info: MatchInfo
    outcome_prediction: OutcomePrediction
    score_prediction: ScorePrediction
    additional_predictions: AdditionalPredictions
    key_factors: KeyFactors
    fixture_id: Optional[int] = None
    gameweek: Optional[int] = None


class PredictionRequest(BaseModel):
    """Request for match prediction"""
    home_team_id: int = Field(..., description="Home team database ID")
    away_team_id: int = Field(..., description="Away team database ID")
    match_date: Optional[datetime] = Field(None, description="Match date (optional)")


class TeamResponse(BaseModel):
    """Team information response"""
    team_id: int
    team_name: str
    short_name: str
    stadium_name: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    database_connected: bool


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
