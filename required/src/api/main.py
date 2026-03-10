"""FastAPI application - REST API for Premier League predictions"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import logging

from src.api.schemas import (
    MatchPrediction, PredictionRequest, TeamResponse,
    HealthResponse, ErrorResponse
)
from src.prediction.prediction_system import PredictionSystem
from src.data.database import get_session, Team
from src.config import settings

# Setup logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Premier League Prediction API",
    description="API for predicting Premier League match outcomes, scores, and statistics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction system
prediction_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize prediction system on startup"""
    global prediction_system
    try:
        logger.info("Initializing prediction system...")
        prediction_system = PredictionSystem()
        logger.info("Prediction system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction system: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Premier League Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    # Check database connection
    try:
        session = get_session()
        session.execute("SELECT 1")
        session.close()
        db_connected = True
    except:
        db_connected = False
    
    # Check model status
    model_loaded = prediction_system is not None and prediction_system.outcome_model.is_trained
    
    return {
        "status": "healthy" if (db_connected and model_loaded) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "database_connected": db_connected
    }


@app.get("/teams", response_model=List[TeamResponse], tags=["Teams"])
async def get_teams():
    """Get all Premier League teams"""
    try:
        session = get_session()
        teams = session.query(Team).all()
        session.close()
        
        return [
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "short_name": team.short_name,
                "stadium_name": team.stadium_name
            }
            for team in teams
        ]
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch teams")


@app.get("/teams/{team_id}", response_model=TeamResponse, tags=["Teams"])
async def get_team(team_id: int):
    """Get specific team by ID"""
    try:
        session = get_session()
        team = session.query(Team).filter_by(team_id=team_id).first()
        session.close()
        
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        
        return {
            "team_id": team.team_id,
            "team_name": team.team_name,
            "short_name": team.short_name,
            "stadium_name": team.stadium_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching team: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch team")


@app.post("/predict", response_model=MatchPrediction, tags=["Predictions"])
async def predict_match(request: PredictionRequest):
    """
    Predict outcome for a specific match
    
    - **home_team_id**: Home team database ID
    - **away_team_id**: Away team database ID
    - **match_date**: Optional match date (defaults to today)
    """
    if prediction_system is None:
        raise HTTPException(status_code=503, detail="Prediction system not initialized")
    
    try:
        prediction = prediction_system.predict_match(
            home_team_id=request.home_team_id,
            away_team_id=request.away_team_id,
            match_date=request.match_date
        )
        
        return prediction
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction")


@app.get("/predictions/upcoming", response_model=List[MatchPrediction], tags=["Predictions"])
async def get_upcoming_predictions(
    limit: int = Query(default=10, ge=1, le=50, description="Number of fixtures to predict")
):
    """
    Get predictions for upcoming fixtures
    
    - **limit**: Maximum number of fixtures to predict (1-50)
    """
    if prediction_system is None:
        raise HTTPException(status_code=503, detail="Prediction system not initialized")
    
    try:
        predictions = prediction_system.predict_upcoming_fixtures(limit=limit)
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating upcoming predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate predictions")


@app.get("/predictions/match/{home_team_id}/{away_team_id}", 
         response_model=MatchPrediction, 
         tags=["Predictions"])
async def predict_match_by_ids(home_team_id: int, away_team_id: int):
    """
    Predict outcome for match between two teams (path parameters)
    
    - **home_team_id**: Home team database ID
    - **away_team_id**: Away team database ID
    """
    if prediction_system is None:
        raise HTTPException(status_code=503, detail="Prediction system not initialized")
    
    try:
        prediction = prediction_system.predict_match(
            home_team_id=home_team_id,
            away_team_id=away_team_id
        )
        
        return prediction
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
