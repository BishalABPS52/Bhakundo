"""
Football API Integration for Real-time Premier League Data
Uses Football-Data.org API (free tier available)
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# FPL team ID → DB-format team name (2025-26 season)
FPL_TEAM_ID_TO_NAME = {
    1:  'Arsenal FC',
    2:  'Aston Villa FC',
    3:  'Burnley FC',
    4:  'AFC Bournemouth',
    5:  'Brentford FC',
    6:  'Brighton & Hove Albion FC',
    7:  'Chelsea FC',
    8:  'Crystal Palace FC',
    9:  'Everton FC',
    10: 'Fulham FC',
    11: 'Leeds United FC',
    12: 'Liverpool FC',
    13: 'Manchester City FC',
    14: 'Manchester United FC',
    15: 'Newcastle United FC',
    16: 'Nottingham Forest FC',
    17: 'Sunderland AFC',
    18: 'Tottenham Hotspur FC',
    19: 'West Ham United FC',
    20: 'Wolverhampton Wanderers FC',
}


class FootballAPI:
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_API_KEY', '')
        self.base_url = "https://api.football-data.org/v4"
        self.fpl_base_url = "https://fantasy.premierleague.com/api"
        self.headers = {
            'X-Auth-Token': self.api_key
        } if self.api_key else {}
        # Premier League competition ID
        self.competition_id = 2021  # PL
        # Current season 2025-26
        self.season = 2025

        # FPL bootstrap cache (team map + current GW)
        self._fpl_bootstrap_cache: Optional[Dict] = None
        self._fpl_bootstrap_fetched_at: Optional[datetime] = None

        # Load real 2025-26 season data
        self.real_matches_data = self._load_real_season_data()
                # Stadium mapping for Premier League teams
        self.stadiums = {
            'Arsenal FC': 'Emirates Stadium',
            'Aston Villa FC': 'Villa Park',
            'AFC Bournemouth': 'Vitality Stadium',
            'Brentford FC': 'Gtech Community Stadium',
            'Brighton & Hove Albion FC': 'American Express Stadium',
            'Burnley FC': 'Turf Moor',
            'Chelsea FC': 'Stamford Bridge',
            'Crystal Palace FC': 'Selhurst Park',
            'Everton FC': 'Goodison Park',
            'Fulham FC': 'Craven Cottage',
            'Leeds United FC': 'Elland Road',
            'Liverpool FC': 'Anfield',
            'Manchester City FC': 'Etihad Stadium',
            'Manchester United FC': 'Old Trafford',
            'Newcastle United FC': 'St. James\' Park',
            'Nottingham Forest FC': 'The City Ground',
            'Sunderland AFC': 'Stadium of Light',
            'Tottenham Hotspur FC': 'Tottenham Hotspur Stadium',
            'West Ham United FC': 'London Stadium',
            'Wolverhampton Wanderers FC': 'Molineux Stadium'
        }
    
    # ------------------------------------------------------------------ FPL API --

    def _get_fpl_bootstrap(self) -> Optional[Dict]:
        """Fetch and cache FPL bootstrap-static (teams + events). Cache for 10 min."""
        now = datetime.now()
        if (
            self._fpl_bootstrap_cache is not None
            and self._fpl_bootstrap_fetched_at is not None
            and (now - self._fpl_bootstrap_fetched_at).seconds < 600
        ):
            return self._fpl_bootstrap_cache
        try:
            resp = requests.get(
                f"{self.fpl_base_url}/bootstrap-static/",
                timeout=8,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            if resp.status_code == 200:
                data = resp.json()
                self._fpl_bootstrap_cache = data
                self._fpl_bootstrap_fetched_at = now
                print("✅ FPL bootstrap-static fetched")
                return data
        except Exception as e:
            print(f"⚠️ FPL bootstrap error: {e}")
        return None

    def _fpl_team_map(self) -> Dict[int, str]:
        """
        Return FPL team-id → DB team name map.
        Tries bootstrap-static first; falls back to the hardcoded FPL_TEAM_ID_TO_NAME.
        """
        bootstrap = self._get_fpl_bootstrap()
        if bootstrap:
            fpl_id_to_short = {t['id']: t['name'] for t in bootstrap.get('teams', [])}
            # Build reverse lookup: FPL short name → DB full name
            short_to_db = {
                v.replace(' FC', '').replace(' AFC', '').lower(): v
                for v in self.stadiums.keys()
            }
            # Special cases that don't strip cleanly
            short_to_db.update({
                'afc bournemouth': 'AFC Bournemouth',
                'bournemouth': 'AFC Bournemouth',
                "nott'm forest": 'Nottingham Forest FC',
                'nottingham forest': 'Nottingham Forest FC',
                'man city': 'Manchester City FC',
                'man utd': 'Manchester United FC',
                'spurs': 'Tottenham Hotspur FC',
                'tottenham': 'Tottenham Hotspur FC',
                'wolves': 'Wolverhampton Wanderers FC',
                'wolverhampton': 'Wolverhampton Wanderers FC',
                'brighton': 'Brighton & Hove Albion FC',
                'newcastle': 'Newcastle United FC',
                'west ham': 'West Ham United FC',
                'crystal palace': 'Crystal Palace FC',
                'sunderland': 'Sunderland AFC',
            })
            result = {}
            for fpl_id, short_name in fpl_id_to_short.items():
                db_name = short_to_db.get(short_name.lower())
                if db_name is None:
                    # Try stripping 'FC' / 'AFC'
                    cleaned = short_name.lower().replace(' fc', '').replace(' afc', '').strip()
                    db_name = short_to_db.get(cleaned)
                result[fpl_id] = db_name or FPL_TEAM_ID_TO_NAME.get(fpl_id, short_name)
            return result
        return FPL_TEAM_ID_TO_NAME

    def get_fpl_fixtures(self, gameweek: Optional[int] = None) -> List[Dict]:
        """Fetch fixtures from FPL API and return in our standard match format."""
        try:
            url = f"{self.fpl_base_url}/fixtures/"
            params = {}
            if gameweek:
                params['event'] = gameweek
            resp = requests.get(
                url, params=params, timeout=8,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            if resp.status_code != 200:
                print(f"⚠️ FPL fixtures API returned {resp.status_code}")
                return []
            raw = resp.json()
            team_map = self._fpl_team_map()
            matches = []
            for m in raw:
                home_id = m.get('team_h')
                away_id = m.get('team_a')
                home = team_map.get(home_id, f'Team {home_id}')
                away  = team_map.get(away_id, f'Team {away_id}')
                gw   = m.get('event')
                fpl_id = m.get('id')
                finished = m.get('finished', False)
                started  = m.get('started', False)
                h_score  = m.get('team_h_score')
                a_score  = m.get('team_a_score')
                if finished:
                    status = 'FINISHED'
                elif started:
                    status = 'IN_PLAY'
                else:
                    status = 'SCHEDULED'
                matches.append({
                    'id': fpl_id,
                    'match_id': f'fpl_{fpl_id}',  # will be overridden by canonical resolution
                    'gameweek': gw,
                    'date': m.get('kickoff_time'),
                    'home_team': home,
                    'away_team': away,
                    'home_score': h_score,
                    'away_score': a_score,
                    'status': status,
                    'venue': self.stadiums.get(home, 'Stadium'),
                })
            print(f"✅ FPL fixtures: {len(matches)} matches for GW{gameweek}")
            return matches
        except Exception as e:
            print(f"⚠️ FPL fixtures error: {e}")
            return []

    def get_fpl_current_gameweek(self) -> Optional[int]:
        """Return the current/upcoming GW from FPL bootstrap-static."""
        bootstrap = self._get_fpl_bootstrap()
        if bootstrap:
            for event in bootstrap.get('events', []):
                if event.get('is_next'):
                    return event['id']
            for event in bootstrap.get('events', []):
                if event.get('is_current'):
                    return event['id']
        return None

    # --------------------------------------------------------- end FPL API block --

    def _load_real_season_data(self) -> Dict:
        """Load real 2025-26 season data from saved API response"""
        
        try:
            data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'formatted_2025_26_matches.json'
            if data_path.exists():
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Loaded real 2025-26 season data: {len(data)} gameweeks")
                    return data
        except Exception as e:
            print(f"⚠️ Could not load real season data: {e}")
        return {}
        
    def get_season_info(self) -> Dict:
        """
        Fetch live season info. Priority: FPL bootstrap → football-data.org → config.
        Returns currentMatchday (the upcoming/active gameweek).
        """
        # 1️⃣ FPL bootstrap-static (free, no auth)
        fpl_gw = self.get_fpl_current_gameweek()
        if fpl_gw:
            return {'current_matchday': fpl_gw, 'source': 'fpl'}

        # 2️⃣ football-data.org
        try:
            url = f"{self.base_url}/competitions/{self.competition_id}"
            params = {'season': self.season}
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_matchday = (
                    data.get('currentSeason', {}).get('currentMatchday') or
                    data.get('currentMatchday')
                )
                if current_matchday:
                    return {'current_matchday': int(current_matchday), 'source': 'api'}
        except Exception as e:
            print(f"⚠️ Season info football-data.org error: {e}")

        # 3️⃣ Config fallback
        return {
            'current_matchday': self.get_current_gameweek(),
            'source': 'config'
        }

    def get_matches(self, status: Optional[str] = None, gameweek: Optional[int] = None) -> List[Dict]:
        """
        Get matches. Priority: FPL API → football-data.org → CSV fallback.
        status: 'SCHEDULED', 'LIVE', 'IN_PLAY', 'PAUSED', 'FINISHED'
        """
        # 1️⃣ Try FPL API first (free, no auth, reliably accessible)
        matches = self.get_fpl_fixtures(gameweek=gameweek)
        if matches:
            if status:
                matches = [m for m in matches if m.get('status') == status]
            return matches

        # 2️⃣ Try football-data.org
        try:
            url = f"{self.base_url}/competitions/{self.competition_id}/matches"
            params = {'season': self.season}
            if status:
                params['status'] = status
            if gameweek:
                params['matchday'] = gameweek
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return self._format_matches(data.get('matches', []))
            elif response.status_code == 429:
                print("⚠️ football-data.org rate limit, using CSV fallback")
            else:
                print(f"⚠️ football-data.org status {response.status_code}, using CSV fallback")
        except Exception as e:
            print(f"⚠️ football-data.org error: {e}, using CSV fallback")

        # 3️⃣ CSV fallback
        return self._get_fallback_data(gameweek, status)
    
    def _format_matches(self, matches: List[Dict]) -> List[Dict]:
        """Format API response to our structure"""
        formatted = []
        for match in matches:
            home_team = match.get('homeTeam', {}).get('name', '')
            api_id = match.get('id')
            formatted.append({
                'id': api_id,
                'match_id': str(api_id) if api_id else None,
                'gameweek': match.get('matchday'),
                'date': match.get('utcDate'),
                'home_team': home_team,
                'away_team': match.get('awayTeam', {}).get('name', ''),
                'home_score': match.get('score', {}).get('fullTime', {}).get('home'),
                'away_score': match.get('score', {}).get('fullTime', {}).get('away'),
                'status': match.get('status', 'SCHEDULED'),
                'venue': self.stadiums.get(home_team, 'Stadium')
            })
        return formatted
    
    def _get_fallback_data(self, gameweek: Optional[int] = None, status: Optional[str] = None) -> List[Dict]:
        """
        Use real 2025-26 season data loaded from API response
        """
        if not self.real_matches_data:
            print("⚠️ No real season data available")
            return []
        
        matches = []
        
        # Filter by gameweek if specified
        if gameweek:
            gw_str = str(gameweek)
            if gw_str in self.real_matches_data:
                matches = self.real_matches_data[gw_str]
        else:
            # Get all matches
            for gw in sorted(self.real_matches_data.keys(), key=int):
                matches.extend(self.real_matches_data[gw])
        
        # Add venue information
        for match in matches:
            home_team = match.get('home_team', '')
            if 'venue' not in match:
                match['venue'] = self.stadiums.get(home_team, 'Stadium')
        
        # Filter by status if specified
        if status:
            matches = [m for m in matches if m.get('status') == status]
        
        return matches
    
    def get_current_gameweek(self) -> int:
        """Get current gameweek number"""
        # Check config file — try both paths (required/data and data)
        base = Path(__file__).parent.parent
        for config_path in [base / 'required' / 'data' / 'config.json', base / 'data' / 'config.json']:
            try:
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        return config.get('current_gameweek', 30)
            except Exception:
                pass
        # Default to gameweek 30 for 2025-26 season (March 2026)
        return 30
    
    def get_standings(self) -> List[Dict]:
        """Get current Premier League standings"""
        try:
            url = f"{self.base_url}/competitions/{self.competition_id}/standings"
            params = {'season': self.season}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                standings = data.get('standings', [])
                if standings:
                    return self._format_standings(standings[0].get('table', []))
            
            return self._get_fallback_standings()
                
        except Exception as e:
            print(f"⚠️ Standings API error: {e}")
            return self._get_fallback_standings()
    
    def _format_standings(self, table: List[Dict]) -> List[Dict]:
        """Format standings data"""
        formatted = []
        for entry in table:
            formatted.append({
                'position': entry.get('position'),
                'team': entry.get('team', {}).get('name', ''),
                'played': entry.get('playedGames'),
                'won': entry.get('won'),
                'drawn': entry.get('draw'),
                'lost': entry.get('lost'),
                'gf': entry.get('goalsFor'),
                'ga': entry.get('goalsAgainst'),
                'gd': entry.get('goalDifference'),
                'points': entry.get('points'),
                'form': entry.get('form', '')
            })
        return formatted
    
    def _get_fallback_standings(self) -> List[Dict]:
        """Fallback standings data - Official 2025-26 Premier League standings after GW20"""
        return [
            {'position': 1, 'team': 'Arsenal FC', 'played': 20, 'won': 15, 'drawn': 3, 'lost': 2, 'gf': 40, 'ga': 14, 'gd': 26, 'points': 48, 'form': 'W,W,W,W,W'},
            {'position': 2, 'team': 'Manchester City FC', 'played': 20, 'won': 13, 'drawn': 3, 'lost': 4, 'gf': 44, 'ga': 18, 'gd': 26, 'points': 42, 'form': 'W,W,W,D,D'},
            {'position': 3, 'team': 'Aston Villa FC', 'played': 20, 'won': 13, 'drawn': 3, 'lost': 4, 'gf': 33, 'ga': 24, 'gd': 9, 'points': 42, 'form': 'W,W,W,L,W'},
            {'position': 4, 'team': 'Liverpool FC', 'played': 20, 'won': 10, 'drawn': 4, 'lost': 6, 'gf': 32, 'ga': 28, 'gd': 4, 'points': 34, 'form': 'W,W,W,D,D'},
            {'position': 5, 'team': 'Chelsea FC', 'played': 20, 'won': 8, 'drawn': 7, 'lost': 5, 'gf': 33, 'ga': 22, 'gd': 11, 'points': 31, 'form': 'D,W,D,L,D'},
            {'position': 6, 'team': 'Manchester United FC', 'played': 20, 'won': 8, 'drawn': 7, 'lost': 5, 'gf': 34, 'ga': 30, 'gd': 4, 'points': 31, 'form': 'D,L,W,D,D'},
            {'position': 7, 'team': 'Brentford FC', 'played': 20, 'won': 9, 'drawn': 3, 'lost': 8, 'gf': 32, 'ga': 28, 'gd': 4, 'points': 30, 'form': 'D,W,W,D,W'},
            {'position': 8, 'team': 'Sunderland AFC', 'played': 20, 'won': 7, 'drawn': 9, 'lost': 4, 'gf': 21, 'ga': 19, 'gd': 2, 'points': 30, 'form': 'W,D,D,D,D'},
            {'position': 9, 'team': 'Newcastle United FC', 'played': 20, 'won': 8, 'drawn': 5, 'lost': 7, 'gf': 28, 'ga': 24, 'gd': 4, 'points': 29, 'form': 'D,L,W,W,W'},
            {'position': 10, 'team': 'Brighton & Hove Albion FC', 'played': 20, 'won': 7, 'drawn': 7, 'lost': 6, 'gf': 30, 'ga': 27, 'gd': 3, 'points': 28, 'form': 'L,D,L,D,W'},
            {'position': 11, 'team': 'Fulham FC', 'played': 20, 'won': 8, 'drawn': 4, 'lost': 8, 'gf': 28, 'ga': 29, 'gd': -1, 'points': 28, 'form': 'W,W,W,D,D'},
            {'position': 12, 'team': 'Everton FC', 'played': 20, 'won': 8, 'drawn': 4, 'lost': 8, 'gf': 22, 'ga': 24, 'gd': -2, 'points': 28, 'form': 'L,L,D,W,L'},
            {'position': 13, 'team': 'Tottenham Hotspur FC', 'played': 20, 'won': 7, 'drawn': 6, 'lost': 7, 'gf': 28, 'ga': 24, 'gd': 4, 'points': 27, 'form': 'L,L,W,D,D'},
            {'position': 14, 'team': 'Crystal Palace FC', 'played': 20, 'won': 7, 'drawn': 6, 'lost': 7, 'gf': 22, 'ga': 23, 'gd': -1, 'points': 27, 'form': 'L,L,L,D,L'},
            {'position': 15, 'team': 'AFC Bournemouth', 'played': 20, 'won': 5, 'drawn': 8, 'lost': 7, 'gf': 31, 'ga': 38, 'gd': -7, 'points': 23, 'form': 'D,D,L,D,L'},
            {'position': 16, 'team': 'Leeds United FC', 'played': 20, 'won': 5, 'drawn': 7, 'lost': 8, 'gf': 26, 'ga': 33, 'gd': -7, 'points': 22, 'form': 'D,W,D,D,D'},
            {'position': 17, 'team': 'Nottingham Forest FC', 'played': 20, 'won': 5, 'drawn': 3, 'lost': 12, 'gf': 19, 'ga': 33, 'gd': -14, 'points': 18, 'form': 'W,L,L,L,L'},
            {'position': 18, 'team': 'West Ham United FC', 'played': 20, 'won': 3, 'drawn': 5, 'lost': 12, 'gf': 21, 'ga': 41, 'gd': -20, 'points': 14, 'form': 'L,L,L,D,L'},
            {'position': 19, 'team': 'Burnley FC', 'played': 20, 'won': 3, 'drawn': 3, 'lost': 14, 'gf': 20, 'ga': 39, 'gd': -19, 'points': 12, 'form': 'L,D,D,L,L'},
            {'position': 20, 'team': 'Wolverhampton Wanderers FC', 'played': 20, 'won': 1, 'drawn': 3, 'lost': 16, 'gf': 14, 'ga': 40, 'gd': -26, 'points': 6, 'form': 'L,L,L,D,W'}
        ]

    def sync_actual_from_api(self, db, gameweeks: list = None) -> dict:
        """
        Fetch finished matches from the Football API and upsert into the actual table.
        Matches existing rows by (home_team, away_team, gameweek) — safe regardless of match_id source.
        When inserting a new Actual row, the canonical match_id is resolved in this priority:
          1. Existing Actual row's match_id (update path)
          2. Prediction table's match_id for the same (home, away, gw) — keeps alignment with saved predictions
          3. Football API's numeric match id
          4. Fallback: "{gw}_{home[:3]}_{away[:3]}"
        """
        from backend.database import Actual, Prediction

        if gameweeks is None:
            current = self.get_current_gameweek()
            gameweeks = list(range(max(1, current - 3), current + 2))

        updated = inserted = errors = 0

        for gw in gameweeks:
            try:
                matches = self.get_matches(gameweek=gw)
                for match in matches:
                    status = match.get('status', '')
                    if status not in ('FINISHED', 'COMPLETE', 'FT'):
                        continue

                    home   = match.get('home_team')
                    away   = match.get('away_team')
                    # Both field-name variants (_format_matches → home_score; fallback data → home_goals)
                    h_goals = match.get('home_goals') if match.get('home_goals') is not None else match.get('home_score')
                    a_goals = match.get('away_goals') if match.get('away_goals') is not None else match.get('away_score')

                    if not home or not away or h_goals is None or a_goals is None:
                        continue

                    h_goals, a_goals = int(h_goals), int(a_goals)
                    if h_goals > a_goals:   outcome = 'Home Win'
                    elif a_goals > h_goals: outcome = 'Away Win'
                    else:                   outcome = 'Draw'

                    match_date = None
                    try:
                        raw_date = match.get('date') or match.get('match_date')
                        if raw_date:
                            match_date = datetime.fromisoformat(str(raw_date).replace('Z', '+00:00'))
                    except Exception:
                        pass

                    try:
                        rec = db.query(Actual).filter(
                            Actual.home_team == home,
                            Actual.away_team == away,
                            Actual.gameweek  == gw,
                        ).first()

                        if rec:
                            rec.actual_home_goals = h_goals
                            rec.actual_away_goals = a_goals
                            rec.actual_outcome    = outcome
                            rec.status            = 'FINISHED'
                            rec.last_api_update   = datetime.utcnow()
                            if match_date:
                                rec.match_date = match_date
                            updated += 1
                        else:
                            # Resolve canonical match_id:
                            # prefer Prediction table's id (keeps frontend alignment)
                            pred_row = db.query(Prediction).filter(
                                Prediction.home_team == home,
                                Prediction.away_team == away,
                                Prediction.gameweek  == gw,
                            ).order_by(Prediction.created_at).first()

                            api_id = str(match.get('id', ''))
                            canonical_id = (
                                pred_row.match_id if pred_row and pred_row.match_id
                                else api_id
                                or f"{gw}_{home[:3]}_{away[:3]}"
                            )

                            db.add(Actual(
                                match_id          = canonical_id,
                                home_team         = home,
                                away_team         = away,
                                gameweek          = gw,
                                match_date        = match_date,
                                season            = '2025-26',
                                actual_home_goals = h_goals,
                                actual_away_goals = a_goals,
                                actual_outcome    = outcome,
                                status            = 'FINISHED',
                                last_api_update   = datetime.utcnow(),
                            ))
                            inserted += 1

                    except Exception as row_err:
                        print(f"  ⚠ sync_actual row error GW{gw} {home} vs {away}: {row_err}")
                        errors += 1

            except Exception as gw_err:
                print(f"  ⚠ sync_actual GW{gw} error: {gw_err}")
                errors += 1

        try:
            db.commit()
        except Exception as commit_err:
            db.rollback()
            print(f"  ⚠ sync_actual commit error: {commit_err}")

        return {'updated': updated, 'inserted': inserted, 'errors': errors}


# Create singleton instance
football_api = FootballAPI()
