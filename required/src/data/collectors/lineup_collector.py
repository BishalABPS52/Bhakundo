"""Fetch and store player lineup data from Football-Data.org API"""

import requests
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path

from src.config import settings, FOOTBALL_DATA_BASE_URL

logger = logging.getLogger(__name__)


class LineupDataCollector:
    """Collector for match lineups from Football-Data.org API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.football_data_api_key
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.headers = {"X-Auth-Token": self.api_key}
        self.rate_limit_delay = 6  # 10 requests per minute
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def get_match_details(self, match_id: int) -> Dict:
        """Get detailed match information including lineups"""
        endpoint = f"matches/{match_id}"
        return self._make_request(endpoint)
    
    def get_team_squad(self, team_id: int, season: str = "2024") -> Dict:
        """Get team squad for a season"""
        endpoint = f"teams/{team_id}"
        params = {"season": season}
        return self._make_request(endpoint, params)
    
    def extract_lineup_features(self, match_details: Dict) -> Dict:
        """Extract lineup features from match details"""
        if not match_details or "homeTeam" not in match_details:
            return {}
        
        features = {
            "match_id": match_details.get("id"),
            "match_date": match_details.get("utcDate"),
            "home_team": match_details["homeTeam"]["name"],
            "away_team": match_details["awayTeam"]["name"],
        }
        
        # Home team lineup
        if "lineup" in match_details.get("homeTeam", {}):
            home_lineup = match_details["homeTeam"]["lineup"]
            features["home_formation"] = home_lineup.get("formation", "Unknown")
            features["home_lineup_players"] = []
            
            if "startXI" in home_lineup:
                for player in home_lineup["startXI"]:
                    features["home_lineup_players"].append({
                        "name": player["name"],
                        "position": player.get("position"),
                        "shirt_number": player.get("shirtNumber")
                    })
        
        # Away team lineup
        if "lineup" in match_details.get("awayTeam", {}):
            away_lineup = match_details["awayTeam"]["lineup"]
            features["away_formation"] = away_lineup.get("formation", "Unknown")
            features["away_lineup_players"] = []
            
            if "startXI" in away_lineup:
                for player in away_lineup["startXI"]:
                    features["away_lineup_players"].append({
                        "name": player["name"],
                        "position": player.get("position"),
                        "shirt_number": player.get("shirtNumber")
                    })
        
        # Match result
        if "score" in match_details:
            score = match_details["score"]
            features["home_goals"] = score["fullTime"]["home"]
            features["away_goals"] = score["fullTime"]["away"]
        
        return features
    
    def collect_season_lineups(self, season: str = "2024") -> pd.DataFrame:
        """Collect lineups for all matches in a season"""
        logger.info(f"Collecting lineups for season {season}...")
        
        # First get all matches for the season
        endpoint = "competitions/PL/matches"
        params = {"season": season, "status": "FINISHED"}
        
        matches_data = self._make_request(endpoint, params)
        
        if not matches_data or "matches" not in matches_data:
            logger.warning(f"No matches found for season {season}")
            return pd.DataFrame()
        
        lineup_data = []
        total_matches = len(matches_data["matches"])
        
        for idx, match in enumerate(matches_data["matches"], 1):
            match_id = match.get("id")
            logger.info(f"Processing match {idx}/{total_matches}: {match_id}")
            
            # Get detailed match info
            match_details = self.get_match_details(match_id)
            
            if match_details:
                lineup_features = self.extract_lineup_features(match_details)
                if lineup_features:
                    lineup_data.append(lineup_features)
        
        # Convert to DataFrame
        if lineup_data:
            df = pd.DataFrame(lineup_data)
            logger.info(f"Collected lineups for {len(df)} matches")
            return df
        else:
            logger.warning("No lineup data collected")
            return pd.DataFrame()
    
    def save_lineups(self, df: pd.DataFrame, filename: str):
        """Save lineup data to JSON (better for nested structures)"""
        output_path = Path(f"data/raw/{filename}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to records for JSON
        data = df.to_dict('records')
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved lineup data to {output_path}")


def main():
    """Collect lineup data for recent seasons"""
    collector = LineupDataCollector()
    
    # Collect for 2024-25 season (current season with most complete data)
    seasons = ["2024", "2023"]
    
    for season in seasons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting lineups for {season}-{str(int(season)+1)[-2:]} season")
        logger.info(f"{'='*60}")
        
        df = collector.collect_season_lineups(season)
        
        if not df.empty:
            filename = f"lineups_{season}.json"
            collector.save_lineups(df, filename)
        else:
            logger.warning(f"No data collected for season {season}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
