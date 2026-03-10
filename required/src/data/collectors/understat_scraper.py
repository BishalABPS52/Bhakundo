"""Understat scraper for xG (Expected Goals) data using direct HTTP"""

import pandas as pd
import time
import re
import json
import requests
from typing import Optional, Dict, List
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

UNDERSTAT_AVAILABLE = True


class UnderstatScraper:
    """Scraper for Understat xG data using direct HTTP requests"""
    
    def __init__(self):
        self.base_url = "https://understat.com"
        self.league_code = "EPL"  # Premier League code
        self.rate_limit_delay = 2  # Seconds between requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _extract_json_data(self, html: str, variable_name: str) -> List[Dict]:
        """Extract JSON data from JavaScript variable in HTML"""
        try:
            # Find the JavaScript variable
            pattern = rf"var {variable_name} = JSON\.parse\('(.+?)'\);"
            match = re.search(pattern, html)
            
            if match:
                json_str = match.group(1)
                # Unescape the string
                json_str = json_str.encode().decode('unicode_escape')
                return json.loads(json_str)
            return []
        except Exception as e:
            logger.error(f"Error extracting {variable_name}: {e}")
            return []
    
    def get_league_matches(self, season: str) -> pd.DataFrame:
        """
        Fetch all Premier League matches with xG data for a season
        
        Args:
            season: Year (e.g., "2023" for 2023-24 season)
            
        Returns:
            DataFrame with xG data
        """
        logger.info(f"Fetching xG data for season {season} from Understat...")
        
        url = f"{self.base_url}/league/{self.league_code}/{season}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            
            # Extract matches data from JavaScript
            matches_data = self._extract_json_data(response.text, 'datesData')
            
            if not matches_data:
                logger.warning(f"No xG data found for season {season}")
                return pd.DataFrame()
            
            # Process matches
            processed_matches = []
            for match in matches_data:
                match_info = {
                    'understat_match_id': match.get('id'),
                    'match_date': match.get('datetime'),
                    'home_team': match.get('h', {}).get('title'),
                    'away_team': match.get('a', {}).get('title'),
                    'home_goals': match.get('goals', {}).get('h'),
                    'away_goals': match.get('goals', {}).get('a'),
                    'home_xg': float(match.get('xG', {}).get('h', 0)),
                    'away_xg': float(match.get('xG', {}).get('a', 0)),
                    'season': season
                }
                processed_matches.append(match_info)
            
            df = pd.DataFrame(processed_matches)
            logger.info(f"✓ Retrieved xG data for {len(df)} matches")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching xG data: {e}")
            return pd.DataFrame()
    
    def get_match_shots(self, match_id: str) -> pd.DataFrame:
        """
        Get detailed shot data for a specific match
        
        Args:
            match_id: Understat match ID
            
        Returns:
            DataFrame with shot-level data including xG
        """
        logger.info(f"Fetching shot data for match {match_id}...")
        
        url = f"{self.base_url}/match/{match_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            
            # Extract shots data
            shots_data = self._extract_json_data(response.text, 'shotsData')
            
            if not shots_data:
                return pd.DataFrame()
            
            # Flatten shots data
            shots = []
            for side in ['h', 'a']:  # home and away
                if side in shots_data:
                    for shot in shots_data[side]:
                        shot_info = {
                            'match_id': match_id,
                            'team_side': 'home' if side == 'h' else 'away',
                            'player': shot.get('player'),
                            'player_id': shot.get('player_id'),
                            'minute': shot.get('minute'),
                            'result': shot.get('result'),
                            'xg': float(shot.get('xG', 0)),
                            'x': float(shot.get('X', 0)),
                            'y': float(shot.get('Y', 0)),
                            'situation': shot.get('situation'),
                            'shot_type': shot.get('shotType'),
                            'last_action': shot.get('lastAction')
                        }
                        shots.append(shot_info)
            
            return pd.DataFrame(shots)
            
        except Exception as e:
            logger.error(f"Error fetching shot data: {e}")
            return pd.DataFrame()
    
    def get_player_xg_data(self, season: str, player_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get player-level xG statistics for a season
        
        Args:
            season: Year (e.g., "2023")
            player_name: Optional player name filter
            
        Returns:
            DataFrame with player xG statistics
        """
        logger.info(f"Fetching player xG data for season {season}...")
        
        url = f"{self.base_url}/league/{self.league_code}/{season}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            
            # Extract players data
            players_data = self._extract_json_data(response.text, 'playersData')
            
            if not players_data:
                return pd.DataFrame()
            
            # Process player stats
            player_stats = []
            for player in players_data:
                stats = {
                    'player_id': player.get('id'),
                    'player_name': player.get('player_name'),
                    'team': player.get('team_title'),
                    'position': player.get('position'),
                    'games': int(player.get('games', 0)),
                    'time': int(player.get('time', 0)),
                    'goals': int(player.get('goals', 0)),
                    'xg': float(player.get('xG', 0)),
                    'assists': int(player.get('assists', 0)),
                    'xa': float(player.get('xA', 0)),
                    'shots': int(player.get('shots', 0)),
                    'key_passes': int(player.get('key_passes', 0)),
                    'npg': int(player.get('npg', 0)),  # Non-penalty goals
                    'npxg': float(player.get('npxG', 0)),  # Non-penalty xG
                    'season': season
                }
                
                # Filter by player name if specified
                if player_name is None or player_name.lower() in stats['player_name'].lower():
                    player_stats.append(stats)
            
            df = pd.DataFrame(player_stats)
            logger.info(f"✓ Retrieved xG data for {len(df)} players")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching player xG data: {e}")
            return pd.DataFrame()
    
    def get_team_xg_stats(self, season: str) -> pd.DataFrame:
        """
        Get aggregated team xG statistics
        
        Args:
            season: Year (e.g., "2023")
            
        Returns:
            DataFrame with team-level xG stats
        """
        logger.info(f"Calculating team xG statistics for season {season}...")
        
        # Get all matches
        matches_df = self.get_league_matches(season)
        
        if matches_df.empty:
            return pd.DataFrame()
        
        # Aggregate by team
        team_stats = []
        
        for team in pd.concat([matches_df['home_team'], matches_df['away_team']]).unique():
            home_matches = matches_df[matches_df['home_team'] == team]
            away_matches = matches_df[matches_df['away_team'] == team]
            
            stats = {
                'team': team,
                'matches_played': len(home_matches) + len(away_matches),
                'xg_for': home_matches['home_xg'].sum() + away_matches['away_xg'].sum(),
                'xg_against': home_matches['away_xg'].sum() + away_matches['home_xg'].sum(),
                'goals_for': home_matches['home_goals'].sum() + away_matches['away_goals'].sum(),
                'goals_against': home_matches['away_goals'].sum() + away_matches['home_goals'].sum(),
                'season': season
            }
            
            # Calculate derived metrics
            stats['xg_diff'] = stats['xg_for'] - stats['xg_against']
            stats['goal_diff'] = stats['goals_for'] - stats['goals_against']
            stats['xg_overperformance'] = stats['goals_for'] - stats['xg_for']
            stats['avg_xg_for'] = stats['xg_for'] / stats['matches_played']
            stats['avg_xg_against'] = stats['xg_against'] / stats['matches_played']
            
            team_stats.append(stats)
        
        df = pd.DataFrame(team_stats)
        logger.info(f"✓ Calculated xG stats for {len(df)} teams")
        return df


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    
    scraper = UnderstatScraper()
    
    # Test getting matches
    matches = scraper.get_league_matches("2023")
    print(f"\nMatches retrieved: {len(matches)}")
    if not matches.empty:
        print(matches.head())
    
    # Test getting team stats
    team_stats = scraper.get_team_xg_stats("2023")
    print(f"\nTeam stats retrieved: {len(team_stats)}")
    if not team_stats.empty:
        print(team_stats.head())
