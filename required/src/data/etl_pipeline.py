"""ETL Pipeline - Orchestrates data collection from all sources"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from pathlib import Path

from src.data.collectors.football_data_api import FootballDataAPI
from src.data.collectors.understat_scraper import UnderstatScraper, UNDERSTAT_AVAILABLE
from src.data.collectors.fpl_api import FPLDataCollector
from src.data.database import get_session, Match, Team, Player, TeamForm, Injury, Fixture
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_COLLECTION_CONFIG

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Main ETL pipeline for collecting and processing football data"""
    
    def __init__(self):
        self.football_data_api = FootballDataAPI()
        self.fpl_api = FPLDataCollector()
        
        if UNDERSTAT_AVAILABLE:
            self.understat = UnderstatScraper()
        else:
            self.understat = None
            logger.warning("Understat scraper not available. xG data will not be collected.")
        
        self.session = get_session()
    
    def _get_team_mapping(self) -> dict:
        """Create mapping of team names to database IDs"""
        teams = self.session.query(Team).all()
        mapping = {}
        
        for team in teams:
            # Map both full name and variations
            mapping[team.team_name.lower()] = team.team_id
            mapping[team.short_name.lower()] = team.team_id
            
            # Add common variations
            variations = {
                "manchester city": ["man city"],
                "manchester united": ["man utd", "man united"],
                "newcastle": ["newcastle united"],
                "west ham": ["west ham united"],
                "nottingham forest": ["nott'm forest"],
                "wolves": ["wolverhampton"],
                "tottenham": ["spurs", "tottenham hotspur"],
                "brighton": ["brighton & hove albion", "brighton and hove albion"],
            }
            
            for full_name, aliases in variations.items():
                if team.team_name.lower() == full_name:
                    for alias in aliases:
                        mapping[alias.lower()] = team.team_id
        
        return mapping
    
    def _find_team_id(self, team_name: str, mapping: dict) -> Optional[int]:
        """Find team ID from name using fuzzy matching"""
        team_lower = team_name.lower().strip()
        return mapping.get(team_lower)
    
    def collect_historical_matches(self, seasons: List[str]) -> pd.DataFrame:
        """
        Collect historical match data from Football-Data.org
        
        Args:
            seasons: List of season years (e.g., ["2019", "2020", "2021"])
        """
        logger.info(f"Collecting historical matches for seasons: {seasons}")
        
        all_matches = []
        
        for season in seasons:
            try:
                # Get matches from Football-Data.org
                matches = self.football_data_api.get_matches(season, status="FINISHED")
                
                if not matches.empty:
                    all_matches.append(matches)
                    logger.info(f"Collected {len(matches)} matches for season {season}")
                
            except Exception as e:
                logger.error(f"Error collecting matches for season {season}: {e}")
        
        if all_matches:
            df = pd.concat(all_matches, ignore_index=True)
            
            # Save raw data
            raw_file = RAW_DATA_DIR / f"historical_matches_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(raw_file, index=False)
            logger.info(f"Saved {len(df)} matches to {raw_file}")
            
            return df
        
        return pd.DataFrame()
    
    def collect_xg_data(self, seasons: List[str]) -> pd.DataFrame:
        """
        Collect xG data from Understat
        
        Args:
            seasons: List of season years
        """
        if not self.understat:
            logger.warning("Skipping xG data collection - Understat not available")
            return pd.DataFrame()
        
        logger.info(f"Collecting xG data for seasons: {seasons}")
        
        all_xg_data = []
        
        for season in seasons:
            try:
                xg_matches = self.understat.get_league_matches(season)
                
                if not xg_matches.empty:
                    xg_matches["season"] = f"{season}-{str(int(season) + 1)[-2:]}"
                    all_xg_data.append(xg_matches)
                    logger.info(f"Collected xG data for {len(xg_matches)} matches in season {season}")
                
            except Exception as e:
                logger.error(f"Error collecting xG data for season {season}: {e}")
        
        if all_xg_data:
            df = pd.concat(all_xg_data, ignore_index=True)
            
            # Save raw data
            raw_file = RAW_DATA_DIR / f"xg_data_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(raw_file, index=False)
            logger.info(f"Saved xG data for {len(df)} matches to {raw_file}")
            
            return df
        
        return pd.DataFrame()
    
    def collect_fpl_data(self) -> dict:
        """Collect current FPL data (players, teams, fixtures, injuries)"""
        logger.info("Collecting FPL data...")
        
        data = {}
        
        try:
            # Get players
            data["players"] = self.fpl_api.get_players_data()
            logger.info(f"Collected data for {len(data['players'])} players")
            
            # Get teams
            data["teams"] = self.fpl_api.get_teams_data()
            logger.info(f"Collected data for {len(data['teams'])} teams")
            
            # Get fixtures
            data["fixtures"] = self.fpl_api.get_fixtures()
            logger.info(f"Collected {len(data['fixtures'])} fixtures")
            
            # Get injuries
            data["injuries"] = self.fpl_api.get_injuries_and_suspensions()
            logger.info(f"Collected {len(data['injuries'])} injury/availability updates")
            
            # Save raw data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for key, df in data.items():
                if not df.empty:
                    raw_file = RAW_DATA_DIR / f"fpl_{key}_{timestamp}.csv"
                    df.to_csv(raw_file, index=False)
            
        except Exception as e:
            logger.error(f"Error collecting FPL data: {e}")
        
        return data
    
    def merge_match_data(self, matches_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge match data from different sources
        
        Args:
            matches_df: Matches from Football-Data.org
            xg_df: xG data from Understat
        """
        logger.info("Merging match data from multiple sources...")
        
        if matches_df.empty:
            return pd.DataFrame()
        
        # Convert dates to datetime
        matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
        
        if not xg_df.empty and "date" in xg_df.columns:
            xg_df["date"] = pd.to_datetime(xg_df["date"])
            
            # Merge on date and team names
            merged = matches_df.merge(
                xg_df[["date", "home_team", "away_team", "home_xg", "away_xg"]],
                left_on=["match_date", "home_team", "away_team"],
                right_on=["date", "home_team", "away_team"],
                how="left",
                suffixes=("", "_understat")
            )
            
            # Use xG from Understat if available
            if "home_xg_understat" in merged.columns:
                merged["home_xg"] = merged["home_xg_understat"].fillna(merged.get("home_xg", 0))
                merged["away_xg"] = merged["away_xg_understat"].fillna(merged.get("away_xg", 0))
            
            merged = merged.drop(columns=[c for c in merged.columns if "_understat" in c or c == "date"])
            
            logger.info(f"Merged data: {len(merged)} matches with xG data")
            return merged
        
        return matches_df
    
    def load_to_database(self, matches_df: pd.DataFrame):
        """Load processed match data to database"""
        logger.info("Loading match data to database...")
        
        if matches_df.empty:
            logger.warning("No matches to load")
            return
        
        team_mapping = self._get_team_mapping()
        loaded_count = 0
        
        for _, row in matches_df.iterrows():
            try:
                # Find team IDs
                home_team_id = self._find_team_id(row["home_team"], team_mapping)
                away_team_id = self._find_team_id(row["away_team"], team_mapping)
                
                if not home_team_id or not away_team_id:
                    logger.warning(f"Could not find team IDs for {row['home_team']} vs {row['away_team']}")
                    continue
                
                # Check if match already exists
                existing = self.session.query(Match).filter_by(
                    season=row["season"],
                    gameweek=row["gameweek"],
                    home_team_id=home_team_id,
                    away_team_id=away_team_id
                ).first()
                
                if existing:
                    # Update existing match
                    for col in ["home_goals", "away_goals", "result", "home_xg", "away_xg"]:
                        if col in row and pd.notna(row[col]):
                            setattr(existing, col, row[col])
                else:
                    # Create new match
                    match = Match(
                        season=row["season"],
                        gameweek=row["gameweek"],
                        match_date=pd.to_datetime(row["match_date"]),
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        home_goals=row.get("home_goals"),
                        away_goals=row.get("away_goals"),
                        result=row.get("result"),
                        home_xg=row.get("home_xg"),
                        away_xg=row.get("away_xg"),
                        half_time_home_goals=row.get("half_time_home_goals"),
                        half_time_away_goals=row.get("half_time_away_goals"),
                    )
                    self.session.add(match)
                
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Error loading match: {e}")
                continue
        
        try:
            self.session.commit()
            logger.info(f"Successfully loaded {loaded_count} matches to database")
        except Exception as e:
            logger.error(f"Error committing to database: {e}")
            self.session.rollback()
    
    def run_full_pipeline(self, seasons: Optional[List[str]] = None):
        """
        Run complete ETL pipeline
        
        Args:
            seasons: List of seasons to collect (defaults to config)
        """
        if not seasons:
            seasons = DATA_COLLECTION_CONFIG["seasons"]
        
        logger.info("=" * 60)
        logger.info("Starting Full ETL Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Collect historical matches
        matches_df = self.collect_historical_matches(seasons)
        
        # Step 2: Collect xG data
        xg_df = self.collect_xg_data(seasons)
        
        # Step 3: Merge data
        merged_df = self.merge_match_data(matches_df, xg_df)
        
        # Step 4: Load to database
        if not merged_df.empty:
            self.load_to_database(merged_df)
        
        # Step 5: Collect FPL data
        fpl_data = self.collect_fpl_data()
        
        logger.info("=" * 60)
        logger.info("ETL Pipeline Completed")
        logger.info("=" * 60)
    
    def __del__(self):
        """Close database session"""
        if hasattr(self, "session"):
            self.session.close()


if __name__ == "__main__":
    # Test ETL pipeline
    logging.basicConfig(level=logging.INFO)
    
    pipeline = ETLPipeline()
    pipeline.run_full_pipeline(seasons=["2023", "2024"])
