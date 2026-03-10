#!/usr/bin/env python3
"""
Fetch New Gameweek Results Script
Fetches the latest Premier League match results, updates the dataset,
then retrains all models and regenerates predictions in the database.
"""

import os
import sys
import subprocess
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Set up paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "required" / "data" / "raw" / "pl"

# Load environment variables from backend/.env
load_dotenv(BACKEND_DIR / ".env")

# API Configuration
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
FOOTBALL_API_BASE = "https://api.football-data.org/v4"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_latest_pl_matches():
    """Fetch all matches from the current PL season"""
    print("\n" + "="*80)
    print("FETCHING LATEST PREMIER LEAGUE GAMEWEEK DATA")
    print("="*80 + "\n")
    
    if not FOOTBALL_API_KEY:
        print("❌ ERROR: FOOTBALL_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Premier League ID: 2021
    url = f"{FOOTBALL_API_BASE}/competitions/2021/matches"
    headers = {"X-Auth-Token": FOOTBALL_API_KEY}
    
    try:
        print(f"🌐 Fetching from: {url}")
        response = requests.get(url, headers=headers, params={"season": 2025})
        response.raise_for_status()
        data = response.json()
        
        matches = data.get("matches", [])
        print(f"✅ Fetched {len(matches)} total matches from API")
        
        # Process matches
        processed_matches = []
        for match in matches:
            if match["status"] == "FINISHED":
                processed_matches.append({
                    "match_date": match["utcDate"],
                    "season": "2025-26",
                    "gameweek": match["matchday"],
                    "home_team": match["homeTeam"]["name"],
                    "away_team": match["awayTeam"]["name"],
                    "home_score": float(match["score"]["fullTime"]["home"]) if match["score"]["fullTime"]["home"] is not None else 0.0,
                    "away_score": float(match["score"]["fullTime"]["away"]) if match["score"]["fullTime"]["away"] is not None else 0.0,
                    "home_goals": float(match["score"]["fullTime"]["home"]) if match["score"]["fullTime"]["home"] is not None else 0.0,
                    "away_goals": float(match["score"]["fullTime"]["away"]) if match["score"]["fullTime"]["away"] is not None else 0.0,
                    "status": match["status"],
                    "match_id": match["id"]
                })
        
        # Determine result
        for m in processed_matches:
            if m["home_goals"] > m["away_goals"]:
                m["result"] = "H"
            elif m["home_goals"] < m["away_goals"]:
                m["result"] = "A"
            else:
                m["result"] = "D"
        
        df = pd.DataFrame(processed_matches)
        
        # Save to CSV
        output_file = DATA_DIR / "pl_2025_26_completed_matches.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n💾 Saved {len(df)} completed matches to:")
        print(f"   {output_file}")
        
        # Display statistics
        print(f"\n📊 MATCH STATISTICS:")
        print(f"   Total matches: {len(df)}")
        print(f"   Gameweeks completed: {df['gameweek'].min()} - {df['gameweek'].max()}")
        print(f"\n📋 Results breakdown:")
        result_counts = df['result'].value_counts()
        print(f"   Home wins: {result_counts.get('H', 0)}")
        print(f"   Draws: {result_counts.get('D', 0)}")
        print(f"   Away wins: {result_counts.get('A', 0)}")
        
        # Show latest gameweek
        latest_gw = df['gameweek'].max()
        latest_matches = df[df['gameweek'] == latest_gw].sort_values('match_date')
        print(f"\n🆕 Latest Gameweek {latest_gw} matches:")
        for _, match in latest_matches.iterrows():
            date_obj = pd.to_datetime(match['match_date'])
            print(f"   {date_obj.strftime('%Y-%m-%d')} | {match['home_team']:30s} {int(match['home_goals'])}-{int(match['away_goals'])} {match['away_team']}")
        
        print(f"\n✅ Data update complete!")
        print("="*80 + "\n")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        sys.exit(1)


def retrain_models():
    """Retrain all models with the newly fetched data"""
    print("\n" + "="*80)
    print("🤖 RETRAINING MODELS WITH UPDATED DATA")
    print("="*80)
    train_script = SCRIPT_DIR / "train_whole_model.py"
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return False
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(SCRIPT_DIR),
        capture_output=False  # show live output
    )
    if result.returncode != 0:
        print("❌ Model training failed!")
        return False
    print("✅ Models retrained successfully!")
    return True


def regenerate_predictions():
    """Regenerate all predictions in the database"""
    print("\n" + "="*80)
    print("🎯 REGENERATING PREDICTIONS IN DATABASE")
    print("="*80)
    gen_script = BACKEND_DIR / "generate_predictions_improved.py"
    if not gen_script.exists():
        print(f"❌ Generation script not found: {gen_script}")
        return False
    result = subprocess.run(
        [sys.executable, str(gen_script)],
        cwd=str(BACKEND_DIR),
        capture_output=False  # show live output
    )
    if result.returncode != 0:
        print("❌ Prediction generation failed!")
        return False
    print("✅ Predictions regenerated successfully!")
    return True


def main():
    """Main execution"""
    print(f"\n🚀 Starting data fetch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = fetch_latest_pl_matches()
    
    print(f"\n✅ SUCCESS! Fetched {len(df)} completed matches")

    # Step 2: Retrain all models
    print("\n💡 Retraining models with updated data...")
    retrain_ok = retrain_models()

    if not retrain_ok:
        print("⚠️  Skipping prediction regeneration due to training failure.")
        return

    # Step 3: Regenerate predictions
    print("\n💡 Regenerating predictions with new models...")
    regenerate_predictions()

    print("\n" + "="*80)
    print("✅ ALL DONE! Data fetched → Models retrained → Predictions updated")
    print("="*80)


if __name__ == "__main__":
    main()
