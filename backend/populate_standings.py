#!/usr/bin/env python3
"""
Populate the standings table in the database with current season data
Fetches from football-data.org API and stores in DB
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from backend.database import Standing, get_db_session, init_db
from backend.football_api import FootballAPI
from datetime import datetime

def populate_standings():
    """Fetch standings from API and populate database"""
    
    # Initialize database
    print("📊 Initializing database...")
    init_db()
    
    # Get standings from API
    print("🔄 Fetching standings from football-data.org API...")
    api = FootballAPI()
    standings_data = api.get_standings()
    
    if not standings_data:
        print("❌ Failed to fetch standings from API")
        return False
    
    print(f"✅ Retrieved {len(standings_data)} teams")
    
    # Get current gameweek
    current_gw = api.get_current_gameweek()
    print(f"📍 Current Gameweek: {current_gw}")
    
    # Get database session
    db = get_db_session()
    
    try:
        # Delete existing standings for this season and gameweek
        print(f"🗑️  Clearing existing standings for GW{current_gw}...")
        db.query(Standing).filter(
            Standing.season == '2025-26',
            Standing.gameweek == current_gw
        ).delete()
        db.commit()
        
        # Insert new standings
        print(f"💾 Inserting {len(standings_data)} teams into database...")
        for team_data in standings_data:
            standing = Standing(
                season='2025-26',
                gameweek=current_gw,
                team=team_data.get('team'),
                position=team_data.get('position'),
                played=team_data.get('played', 0),
                won=team_data.get('won', 0),
                drawn=team_data.get('drawn', 0),
                lost=team_data.get('lost', 0),
                goals_for=team_data.get('goals_for', team_data.get('gf', 0)),
                goals_against=team_data.get('goals_against', team_data.get('ga', 0)),
                goal_difference=team_data.get('goal_difference', team_data.get('gd', 0)),
                points=team_data.get('points', 0),
                form=team_data.get('form', ''),
                updated_at=datetime.utcnow(),
                source='api'
            )
            db.add(standing)
        
        db.commit()
        print(f"✅ Successfully populated standings table with {len(standings_data)} teams")
        
        # Show summary
        print(f"\n{'='*80}")
        print(f"STANDINGS SUMMARY (GW{current_gw})".center(80))
        print(f"{'='*80}")
        
        stats = db.query(Standing).filter(
            Standing.season == '2025-26',
            Standing.gameweek == current_gw
        ).order_by(Standing.position).all()
        
        print(f"{'Pos':<4} {'Team':<30} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GD':<4} {'Pts':<4} {'Form':<20}")
        print("-" * 80)
        
        for s in stats:
            print(f"{s.position:<4} {s.team:<30} {s.played:<3} {s.won:<3} {s.drawn:<3} {s.lost:<3} "
                  f"{s.goal_difference:<4} {s.points:<4} {s.form:<20}")
        
        print(f"{'='*80}")
        print(f"\n✅ Database updated successfully!")
        print(f"   - Standings table: {len(stats)} teams")
        print(f"   - Season: 2025-26")
        print(f"   - Gameweek: {current_gw}")
        print(f"   - Timestamp: {datetime.utcnow().isoformat()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error populating standings: {e}")
        db.rollback()
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == '__main__':
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    
    success = populate_standings()
    sys.exit(0 if success else 1)
