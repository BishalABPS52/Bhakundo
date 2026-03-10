"""FPL (Fantasy Premier League) API collector"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import logging

from src.config import FPL_BASE_URL

logger = logging.getLogger(__name__)


class FPLDataCollector:
    """Collector for Fantasy Premier League API"""
    
    def __init__(self):
        self.base_url = FPL_BASE_URL
        
    def _make_request(self, endpoint: str) -> Dict:
        """Make API request to FPL"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FPL API request failed: {e}")
            return {}
    
    def get_bootstrap_data(self) -> Dict:
        """
        Get bootstrap-static data (all players, teams, gameweeks)
        This is the main FPL endpoint with comprehensive data
        """
        logger.info("Fetching FPL bootstrap data...")
        data = self._make_request("bootstrap-static/")
        
        if data:
            logger.info("Successfully retrieved FPL bootstrap data")
        
        return data
    
    def get_players_data(self) -> pd.DataFrame:
        """Get all player data from FPL"""
        logger.info("Fetching FPL player data...")
        
        data = self.get_bootstrap_data()
        
        if not data or "elements" not in data:
            return pd.DataFrame()
        
        players = []
        for player in data["elements"]:
            players.append({
                "fpl_id": player["id"],
                "player_name": player["web_name"],
                "full_name": player["first_name"] + " " + player["second_name"],
                "team_code": player["team_code"],
                "position": player["element_type"],  # 1=GK, 2=DEF, 3=MID, 4=FWD
                "price": player["now_cost"] / 10,  # Convert from 0.1m units
                "selected_by_percent": float(player["selected_by_percent"]),
                "form": float(player["form"]) if player["form"] else 0,
                "points_per_game": float(player["points_per_game"]) if player["points_per_game"] else 0,
                "total_points": player["total_points"],
                "goals_scored": player["goals_scored"],
                "assists": player["assists"],
                "clean_sheets": player["clean_sheets"],
                "minutes": player["minutes"],
                "yellow_cards": player["yellow_cards"],
                "red_cards": player["red_cards"],
                "bonus": player["bonus"],
                "influence": float(player["influence"]) if player["influence"] else 0,
                "creativity": float(player["creativity"]) if player["creativity"] else 0,
                "threat": float(player["threat"]) if player["threat"] else 0,
                "ict_index": float(player["ict_index"]) if player["ict_index"] else 0,
                "status": player["status"],  # 'a' = available, 'd' = doubtful, 'i' = injured, etc.
                "news": player["news"],
                "chance_of_playing_next_round": player["chance_of_playing_next_round"],
            })
        
        df = pd.DataFrame(players)
        logger.info(f"Retrieved data for {len(df)} players")
        return df
    
    def get_teams_data(self) -> pd.DataFrame:
        """Get team data including fixture difficulty ratings"""
        logger.info("Fetching FPL team data...")
        
        data = self.get_bootstrap_data()
        
        if not data or "teams" not in data:
            return pd.DataFrame()
        
        teams = []
        for team in data["teams"]:
            teams.append({
                "fpl_team_id": team["id"],
                "team_name": team["name"],
                "short_name": team["short_name"],
                "strength_overall_home": team["strength_overall_home"],
                "strength_overall_away": team["strength_overall_away"],
                "strength_attack_home": team["strength_attack_home"],
                "strength_attack_away": team["strength_attack_away"],
                "strength_defence_home": team["strength_defence_home"],
                "strength_defence_away": team["strength_defence_away"],
                "played": team["played"],
                "win": team["win"],
                "draw": team["draw"],
                "loss": team["loss"],
                "points": team["points"],
                "position": team["position"],
            })
        
        df = pd.DataFrame(teams)
        logger.info(f"Retrieved data for {len(df)} teams")
        return df
    
    def get_fixtures(self) -> pd.DataFrame:
        """Get all fixtures with difficulty ratings"""
        logger.info("Fetching FPL fixtures...")
        
        data = self._make_request("fixtures/")
        
        if not data:
            return pd.DataFrame()
        
        fixtures = []
        for fixture in data:
            fixtures.append({
                "fixture_id": fixture["id"],
                "gameweek": fixture["event"],
                "kickoff_time": fixture["kickoff_time"],
                "home_team_id": fixture["team_h"],
                "away_team_id": fixture["team_a"],
                "home_team_difficulty": fixture["team_h_difficulty"],
                "away_team_difficulty": fixture["team_a_difficulty"],
                "home_score": fixture["team_h_score"],
                "away_score": fixture["team_a_score"],
                "finished": fixture["finished"],
                "started": fixture["started"],
            })
        
        df = pd.DataFrame(fixtures)
        logger.info(f"Retrieved {len(df)} fixtures")
        return df
    
    def get_gameweek_fixtures(self, gameweek: int) -> pd.DataFrame:
        """Get fixtures for a specific gameweek"""
        logger.info(f"Fetching fixtures for gameweek {gameweek}...")
        
        data = self._make_request(f"fixtures/?event={gameweek}")
        
        if not data:
            return pd.DataFrame()
        
        fixtures = []
        for fixture in data:
            fixtures.append({
                "fixture_id": fixture["id"],
                "kickoff_time": fixture["kickoff_time"],
                "home_team_id": fixture["team_h"],
                "away_team_id": fixture["team_a"],
                "home_difficulty": fixture["team_h_difficulty"],
                "away_difficulty": fixture["team_a_difficulty"],
            })
        
        return pd.DataFrame(fixtures)
    
    def get_player_details(self, player_id: int) -> Dict:
        """Get detailed information for a specific player"""
        logger.info(f"Fetching details for player {player_id}...")
        
        data = self._make_request(f"element-summary/{player_id}/")
        return data
    
    def get_player_gameweek_history(self, player_id: int) -> pd.DataFrame:
        """Get gameweek-by-gameweek history for a player"""
        data = self.get_player_details(player_id)
        
        if not data or "history" not in data:
            return pd.DataFrame()
        
        history = []
        for gw in data["history"]:
            history.append({
                "player_id": player_id,
                "gameweek": gw["round"],
                "opponent_team": gw["opponent_team"],
                "was_home": gw["was_home"],
                "total_points": gw["total_points"],
                "minutes": gw["minutes"],
                "goals_scored": gw["goals_scored"],
                "assists": gw["assists"],
                "clean_sheets": gw["clean_sheets"],
                "goals_conceded": gw["goals_conceded"],
                "own_goals": gw["own_goals"],
                "penalties_saved": gw["penalties_saved"],
                "penalties_missed": gw["penalties_missed"],
                "yellow_cards": gw["yellow_cards"],
                "red_cards": gw["red_cards"],
                "saves": gw["saves"],
                "bonus": gw["bonus"],
                "bps": gw["bps"],  # Bonus points system
                "influence": float(gw["influence"]),
                "creativity": float(gw["creativity"]),
                "threat": float(gw["threat"]),
                "ict_index": float(gw["ict_index"]),
                "value": gw["value"],
                "selected": gw["selected"],
            })
        
        return pd.DataFrame(history)
    
    def get_injuries_and_suspensions(self) -> pd.DataFrame:
        """Get current injuries and suspensions from player data"""
        logger.info("Extracting injuries and suspensions...")
        
        players_df = self.get_players_data()
        
        # Filter for players who are not available
        injuries = players_df[
            (players_df["status"] != "a") |  # Not available
            (players_df["chance_of_playing_next_round"].notna()) |
            (players_df["news"] != "")
        ].copy()
        
        injuries = injuries[[
            "fpl_id", "player_name", "full_name", "team_code", 
            "status", "news", "chance_of_playing_next_round"
        ]]
        
        logger.info(f"Found {len(injuries)} players with availability issues")
        return injuries
    
    def get_current_gameweek(self) -> int:
        """Get current gameweek number"""
        data = self.get_bootstrap_data()
        
        if data and "events" in data:
            for event in data["events"]:
                if event["is_current"]:
                    return event["id"]
        
        return 1


if __name__ == "__main__":
    # Test the FPL API
    logging.basicConfig(level=logging.INFO)
    collector = FPLDataCollector()
    
    # Test getting players
    players = collector.get_players_data()
    print(f"\nPlayers shape: {players.shape}")
    print(players.head())
    
    # Test getting teams
    teams = collector.get_teams_data()
    print(f"\nTeams shape: {teams.shape}")
    print(teams.head())
    
    # Test getting fixtures
    fixtures = collector.get_fixtures()
    print(f"\nFixtures shape: {fixtures.shape}")
    print(fixtures.head())
    
    # Test injuries
    injuries = collector.get_injuries_and_suspensions()
    print(f"\nInjuries shape: {injuries.shape}")
    print(injuries.head())
