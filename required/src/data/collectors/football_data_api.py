"""Football-Data.org API collector for Premier League match data"""

import requests
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging

from src.config import settings, FOOTBALL_DATA_BASE_URL

logger = logging.getLogger(__name__)


class FootballDataAPI:
    """Collector for Football-Data.org API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.football_data_api_key
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.headers = {"X-Auth-Token": self.api_key}
        self.rate_limit_delay = 6  # 10 requests per minute = 6 seconds between requests
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def get_matches(self, season: str, status: str = "FINISHED") -> pd.DataFrame:
        """
        Fetch all Premier League matches for a season
        
        Args:
            season: Year (e.g., "2023" for 2023-24 season)
            status: Match status - "FINISHED", "SCHEDULED", or "IN_PLAY"
            
        Returns:
            DataFrame with match data
        """
        logger.info(f"Fetching {status} matches for season {season}...")
        
        endpoint = "competitions/PL/matches"
        params = {"season": season, "status": status}
        
        data = self._make_request(endpoint, params)
        
        if not data or "matches" not in data:
            logger.warning(f"No matches found for season {season}")
            return pd.DataFrame()
        
        matches = []
        for match in data["matches"]:
            match_data = {
                "api_match_id": match.get("id"),
                "season": f"{season}-{str(int(season) + 1)[-2:]}",
                "gameweek": match.get("matchday"),
                "match_date": match.get("utcDate"),
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_team_api_id": match["homeTeam"]["id"],
                "away_team_api_id": match["awayTeam"]["id"],
                "status": match.get("status"),
            }
            
            # Add scores if match is finished
            if status == "FINISHED" and match.get("score"):
                score = match["score"]
                match_data.update({
                    "home_goals": score["fullTime"]["home"],
                    "away_goals": score["fullTime"]["away"],
                    "half_time_home_goals": score["halfTime"]["home"],
                    "half_time_away_goals": score["halfTime"]["away"],
                })
                
                # Determine result
                if score["fullTime"]["home"] > score["fullTime"]["away"]:
                    match_data["result"] = "H"
                elif score["fullTime"]["home"] < score["fullTime"]["away"]:
                    match_data["result"] = "A"
                else:
                    match_data["result"] = "D"
            
            matches.append(match_data)
        
        df = pd.DataFrame(matches)
        logger.info(f"Retrieved {len(df)} matches")
        return df
    
    def get_standings(self, season: str) -> pd.DataFrame:
        """Get current Premier League standings"""
        logger.info(f"Fetching standings for season {season}...")
        
        endpoint = "competitions/PL/standings"
        params = {"season": season}
        
        data = self._make_request(endpoint, params)
        
        if not data or "standings" not in data:
            return pd.DataFrame()
        
        standings = []
        for entry in data["standings"][0]["table"]:
            standings.append({
                "position": entry["position"],
                "team_name": entry["team"]["name"],
                "team_api_id": entry["team"]["id"],
                "played": entry["playedGames"],
                "won": entry["won"],
                "drawn": entry["draw"],
                "lost": entry["lost"],
                "points": entry["points"],
                "goals_for": entry["goalsFor"],
                "goals_against": entry["goalsAgainst"],
                "goal_difference": entry["goalDifference"],
                "form": entry.get("form"),
            })
        
        df = pd.DataFrame(standings)
        logger.info(f"Retrieved standings with {len(df)} teams")
        return df
    
    def get_team_matches(self, team_id: int, season: str) -> pd.DataFrame:
        """Get all matches for a specific team"""
        logger.info(f"Fetching matches for team {team_id} in season {season}...")
        
        endpoint = f"teams/{team_id}/matches"
        params = {"season": season, "competitions": "PL"}
        
        data = self._make_request(endpoint, params)
        
        if not data or "matches" not in data:
            return pd.DataFrame()
        
        matches = []
        for match in data["matches"]:
            matches.append({
                "match_id": match.get("id"),
                "date": match.get("utcDate"),
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_goals": match["score"]["fullTime"]["home"] if match.get("score") else None,
                "away_goals": match["score"]["fullTime"]["away"] if match.get("score") else None,
                "status": match.get("status"),
            })
        
        return pd.DataFrame(matches)
    
    def get_current_matchday(self) -> int:
        """Get current matchday number"""
        endpoint = "competitions/PL/matches"
        params = {"status": "IN_PLAY,SCHEDULED"}
        
        data = self._make_request(endpoint, params)
        
        if data and "matches" in data and len(data["matches"]) > 0:
            return data["matches"][0].get("matchday", 1)
        return 1
    
    def get_fixtures(self, matchday: Optional[int] = None) -> pd.DataFrame:
        """
        Get upcoming fixtures
        
        Args:
            matchday: Specific matchday (optional)
        """
        logger.info(f"Fetching fixtures for matchday {matchday}...")
        
        endpoint = "competitions/PL/matches"
        params = {"status": "SCHEDULED"}
        
        if matchday:
            params["matchday"] = matchday
        
        data = self._make_request(endpoint, params)
        
        if not data or "matches" not in data:
            return pd.DataFrame()
        
        fixtures = []
        for match in data["matches"]:
            fixtures.append({
                "match_id": match.get("id"),
                "gameweek": match.get("matchday"),
                "match_date": match.get("utcDate"),
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_team_api_id": match["homeTeam"]["id"],
                "away_team_api_id": match["awayTeam"]["id"],
            })
        
        df = pd.DataFrame(fixtures)
        logger.info(f"Retrieved {len(df)} fixtures")
        return df


if __name__ == "__main__":
    # Test the API
    logging.basicConfig(level=logging.INFO)
    api = FootballDataAPI()
    
    # Test fetching matches for current season
    matches = api.get_matches("2024")
    print(f"\nMatches shape: {matches.shape}")
    print(matches.head())
    
    # Test fetching standings
    standings = api.get_standings("2024")
    print(f"\nStandings shape: {standings.shape}")
    print(standings.head())
